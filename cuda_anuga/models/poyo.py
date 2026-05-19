#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANUGA Flood Simulation Benchmark: CPU vs MPI vs GPU
===================================================
Versión optimizada — malla rápida + elevación corregida.

Usage:
    python dana_benchmark.py                         # Run all three modes
    python dana_benchmark.py --run gpu               # GPU only
    python dana_benchmark.py --run mpi,gpu --np 4    # Skip CPU, 4 MPI ranks
    python dana_benchmark.py --resolucion 100        # Finer mesh (100 m^2)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import numpy as np

import anuga
import rasterio
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely import prepared
from scipy.interpolate import RegularGridInterpolator
from anuga.parallel import distribute, myid, numprocs, barrier, finalize

# ─────────────────────────────────────────────────────────
# Parche: Parallel_Inlet.statistics()
# ─────────────────────────────────────────────────────────
import anuga.parallel.parallel_inlet as _pi

_original_statistics = _pi.Parallel_Inlet.statistics


def _safe_statistics(self):
    try:
        return _original_statistics(self)
    except IndexError:
        return "Inlet statistics unavailable (parallel index mismatch)\n"


_pi.Parallel_Inlet.statistics = _safe_statistics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# ================================================================
#  CLASE SIMULACIÓN (Optimizada)
# ================================================================


class SimulacionInundacion:
    """Pipeline completo: malla → elevación → rugosidad → simulación."""

    def __init__(
        self,
        esquina_so: list[float],
        esquina_ne: list[float],
        directorio: str,
        nombre: str = "simulacion",
        epsg: int = 25830,
        resolucion_malla: float = 200.0,
        ruta_mdt: str | None = None,
        ruta_rugosidad: str | None = None,
        manning_defecto: float = 0.030,
        descargar_edificios: bool = True,
        generar_fotogramas: bool = True,
        estrategia_edificios: str = "elevar",
        altura_edificio: float = 50.0,
        simplificar_tolerancia: float = 1.0,
        area_minima_edificio: float = 20.0,
        buffer_fusion: float = 2.0,
        cache_edificios: bool = True,
    ):
        self.log = logging.getLogger(f"Simulacion.{nombre}")

        # Geometría
        x_min, y_min = esquina_so
        x_max, y_max = esquina_ne
        self.x_min, self.y_min = x_min, y_min
        self.x_max, self.y_max = x_max, y_max
        self.poligono_utm = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ]
        self.epsg = epsg

        # Rutas
        self.directorio = os.path.abspath(directorio)
        if myid == 0:
            os.makedirs(self.directorio, exist_ok=True)
        self.nombre = nombre
        self.ruta_msh = os.path.join(self.directorio, f"{nombre}.msh")
        self.ruta_gpkg = os.path.join(self.directorio, f"{nombre}_malla.gpkg")
        self.ruta_cache_edificios = os.path.join(
            self.directorio, f"{nombre}_edificios.gpkg"
        )
        self.ruta_mdt = ruta_mdt
        self.ruta_rugosidad = ruta_rugosidad

        # Parámetros
        self.resolucion_malla = resolucion_malla
        self.manning_defecto = manning_defecto
        self.descargar_edificios = descargar_edificios
        self.generar_fotogramas = generar_fotogramas

        # Optimización edificios
        self.estrategia_edificios = estrategia_edificios
        self.altura_edificio = altura_edificio
        self.simplificar_tolerancia = simplificar_tolerancia
        self.area_minima_edificio = area_minima_edificio
        self.buffer_fusion = buffer_fusion
        self.cache_edificios = cache_edificios

        self._inlets: list[dict] = []
        self._edificios_utm: gpd.GeoDataFrame | None = None

    def agregar_inlet(
        self,
        caudal,
        centro: list[float] | None = None,
        zona: list[list[float]] | None = None,
        lado: float | None = None,
        nombre: str = "",
    ):
        if centro is None and zona is None:
            raise ValueError("Debes indicar 'centro' o 'zona'.")
        if not callable(caudal):
            valor = float(caudal)
            caudal = lambda t, _v=valor: _v
        if centro is not None:
            zona = self._calcular_zona_inlet(centro, lado)

        self._inlets.append(
            {
                "zona": zona,
                "caudal": caudal,
                "nombre": nombre or f"inlet_{len(self._inlets)}",
            }
        )
        if myid == 0:
            self.log.info(f"💧 Inlet registrado: {self._inlets[-1]['nombre']}")

    def _calcular_zona_inlet(
        self, centro: list[float], lado: float | None
    ) -> list[list[float]]:
        if lado is None:
            lado = 5.0 * np.sqrt(self.resolucion_malla)
        mitad = lado / 2.0
        cx, cy = centro
        x0 = max(cx - mitad, self.x_min)
        x1 = min(cx + mitad, self.x_max)
        y0 = max(cy - mitad, self.y_min)
        y1 = min(cy + mitad, self.y_max)
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    def ejecutar(
        self,
        duracion: float = 3 * 3600,
        paso_guardado: float = 60,
        forzar_malla: bool = False,
        modo_gpu: bool = False,
    ):
        if myid == 0:
            if os.path.exists(self.ruta_msh) and not forzar_malla:
                self.log.info(f"♻️  Malla existente: {self.ruta_msh} — se reutiliza.")
            else:
                self._generar_malla()
        barrier()

        domain = self._crear_dominio() if myid == 0 else None

        self.log.info(f"📡 Proceso {myid}/{numprocs}: distribuyendo…")
        domain = distribute(domain)

        if modo_gpu:
            self.log.info(f"🖥️ Activando aceleración GPU en proceso {myid}...")
            domain.set_multiprocessor_mode(2)

        self.log.info(f"✅ Proceso {myid}: subdomain listo.")

        self._aplicar_inlets(domain)
        ntri = domain.number_of_elements

        t0 = time.time()
        self._simular(domain, duracion, paso_guardado)
        wall_time = time.time() - t0

        barrier()
        domain.sww_merge()
        if myid == 0:
            self.log.info(f"💾 Resultado final: {self.nombre}.sww")

        return wall_time, ntri

    def _obtener_edificios(self, poly_geom: Polygon) -> gpd.GeoDataFrame | None:
        if self.cache_edificios and os.path.exists(self.ruta_cache_edificios):
            self.log.info(f"♻️  Edificios desde caché: {self.ruta_cache_edificios}")
            edificios_utm = gpd.read_file(self.ruta_cache_edificios)
        else:
            self.log.info("🏙️ Descargando edificios de OSM…")
            try:
                gdf_utm = gpd.GeoDataFrame(
                    index=[0], crs=f"EPSG:{self.epsg}", geometry=[poly_geom]
                )
                poligono_wgs84 = gdf_utm.to_crs("EPSG:4326").geometry.iloc[0]
                edificios = ox.features_from_polygon(
                    poligono_wgs84, tags={"building": True}
                )
                edificios = edificios[
                    edificios.geometry.type.isin(["Polygon", "MultiPolygon"])
                ]
                edificios_utm = edificios.to_crs(f"EPSG:{self.epsg}")
                margen = poly_geom.buffer(-0.1)
                edificios_utm = edificios_utm[edificios_utm.within(margen)]
                edificios_utm = gpd.GeoDataFrame(
                    geometry=edificios_utm.geometry.values, crs=f"EPSG:{self.epsg}"
                )
            except Exception as e:
                self.log.warning(f"⚠️ Error descargando edificios: {e}")
                return None

            if self.cache_edificios:
                edificios_utm.to_file(self.ruta_cache_edificios, driver="GPKG")

        edificios_utm = edificios_utm[
            edificios_utm.geometry.area >= self.area_minima_edificio
        ]

        if self.simplificar_tolerancia > 0:
            edificios_utm["geometry"] = edificios_utm.geometry.simplify(
                self.simplificar_tolerancia, preserve_topology=True
            )
            edificios_utm = edificios_utm[
                edificios_utm.is_valid & ~edificios_utm.is_empty
            ]

        if self.buffer_fusion > 0:
            merged = unary_union(
                edificios_utm.geometry.buffer(self.buffer_fusion)
            ).buffer(-self.buffer_fusion)
            polys = (
                [merged]
                if merged.geom_type == "Polygon"
                else list(merged.geoms)
                if merged.geom_type == "MultiPolygon"
                else []
            )
            edificios_utm = gpd.GeoDataFrame(geometry=polys, crs=f"EPSG:{self.epsg}")

        return edificios_utm

    def _generar_malla(self):
        self.log.info("🚀 Generando malla…")
        poly_geom = Polygon(self.poligono_utm)
        agujeros = []

        if self.descargar_edificios:
            edificios = self._obtener_edificios(poly_geom)
            if edificios is not None and len(edificios) > 0:
                if self.estrategia_edificios == "troquelar":
                    agujeros = self._gdf_a_agujeros(edificios)
                else:
                    self._edificios_utm = edificios

        boundary_tags = {"sur": [0], "este": [1], "norte": [2], "oeste": [3]}
        anuga.create_pmesh_from_regions(
            self.poligono_utm,
            boundary_tags=boundary_tags,
            maximum_triangle_area=self.resolucion_malla,
            interior_holes=agujeros,
            filename=self.ruta_msh,
        )
        self._exportar_gpkg()

    @staticmethod
    def _gdf_a_agujeros(gdf: gpd.GeoDataFrame) -> list:
        agujeros = []
        for geom in gdf.geometry:
            if geom.geom_type == "Polygon":
                agujeros.append(list(geom.exterior.coords))
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    agujeros.append(list(poly.exterior.coords))
        return agujeros

    def _exportar_gpkg(self):
        try:
            domain = anuga.create_domain_from_file(self.ruta_msh)
            nodos = domain.get_nodes(absolute=True)
            triangulos = domain.get_triangles()
            polys = [
                Polygon([nodos[t[0]], nodos[t[1]], nodos[t[2]], nodos[t[0]]])
                for t in triangulos
            ]
            gdf = gpd.GeoDataFrame(geometry=polys, crs=f"EPSG:{self.epsg}")
            gdf.to_file(self.ruta_gpkg, driver="GPKG")
        except Exception:
            pass

    def _crear_dominio(self) -> anuga.Domain:
        self.log.info("📂 Creando dominio desde malla…")
        domain = anuga.Domain(self.ruta_msh)
        domain.set_name(self.nombre)
        domain.set_datadir(self.directorio)
        domain.set_default_order(2)
        domain.set_minimum_storable_height(0.01)

        if self.ruta_rugosidad and os.path.exists(self.ruta_rugosidad):
            try:
                gdf = gpd.read_file(self.ruta_rugosidad)
                domain.set_quantity("friction", gdf["manning"].values)
            except Exception:
                domain.set_quantity("friction", self.manning_defecto)
        else:
            domain.set_quantity("friction", self.manning_defecto)

        if (
            self.estrategia_edificios == "elevar"
            and self.descargar_edificios
            and self._edificios_utm is None
        ):
            self._edificios_utm = self._obtener_edificios(Polygon(self.poligono_utm))

        if self.ruta_mdt and os.path.exists(self.ruta_mdt):
            interp = self._crear_interpolador_mdt()
            geo = domain.geo_reference
            xll, yll = geo.get_xllcorner(), geo.get_yllcorner()

            if (
                self.estrategia_edificios == "elevar"
                and self._edificios_utm is not None
                and len(self._edificios_utm) > 0
            ):
                edificios_union = unary_union(self._edificios_utm.geometry)
                edificios_prep = prepared.prep(edificios_union)

                def elevacion_con_edificios(x, y):
                    xa, ya = x + xll, y + yll
                    z = interp(np.column_stack([ya, xa]))
                    mascara = np.array(
                        [
                            edificios_prep.contains(Point(xi, yi))
                            for xi, yi in zip(xa, ya)
                        ]
                    )
                    z[mascara] += self.altura_edificio
                    return z

                domain.set_quantity(
                    "elevation", elevacion_con_edificios, location="centroids"
                )
            else:
                domain.set_quantity(
                    "elevation",
                    lambda x, y: interp(np.column_stack([y + yll, x + xll])),
                    location="centroids",
                )
        else:
            domain.set_quantity("elevation", 0.0)

        domain.set_quantity("stage", expression="elevation")
        Br = anuga.Reflective_boundary(domain)
        Bt = anuga.Transmissive_boundary(domain)
        tags = {"sur": Bt, "este": Bt, "norte": Br, "oeste": Br}
        if "interior" in domain.get_boundary_tags():
            tags["interior"] = Br
        domain.set_boundary(tags)

        return domain

    def _crear_interpolador_mdt(self) -> RegularGridInterpolator:
        with rasterio.open(self.ruta_mdt) as src:
            elev = src.read(1).astype(np.float64)
            elev[elev < -999] = np.nan
            x = np.linspace(src.bounds.left, src.bounds.right, src.width)
            y = np.linspace(src.bounds.top, src.bounds.bottom, src.height)
            return RegularGridInterpolator(
                (y[::-1], x), np.flipud(elev), bounds_error=False, fill_value=0
            )

    def _aplicar_inlets(self, domain):
        for inlet in self._inlets:
            anuga.Inlet_operator(domain, inlet["zona"], Q=inlet["caudal"])

    def _simular(self, domain, duracion: float, paso: float):
        if self.generar_fotogramas:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

        dir_fotos = os.path.join(self.directorio, "fotogramas")
        if myid == 0 and self.generar_fotogramas:
            os.makedirs(dir_fotos, exist_ok=True)
        barrier()

        for t in domain.evolve(yieldstep=paso, finaltime=duracion):
            if myid != 0:
                continue

            coords = domain.get_centroid_coordinates()
            stage = domain.get_quantity("stage").get_values(location="centroids")
            elev = domain.get_quantity("elevation").get_values(location="centroids")
            xmom = domain.get_quantity("xmomentum").get_values(location="centroids")
            ymom = domain.get_quantity("ymomentum").get_values(location="centroids")

            h = np.maximum(stage - elev, 0)
            v = np.zeros_like(h)
            mask = h > 0.01
            if np.any(mask):
                v[mask] = np.sqrt(xmom[mask] ** 2 + ymom[mask] ** 2) / h[mask]

            self.log.info(f"🌊 t={t / 60:6.1f} min | calado_max={np.max(h):.2f} m")

            if not self.generar_fotogramas:
                continue

            x, y = coords[:, 0], coords[:, 1]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            fig.suptitle(f"Simulación — t = {t / 60:.1f} min", fontsize=16)

            sc1 = ax1.scatter(x, y, c=h, cmap="Blues", s=1, vmin=0, vmax=3.5)
            ax1.set_title("Calado (m)")
            ax1.set_aspect("equal")
            plt.colorbar(sc1, ax=ax1)

            sc2 = ax2.scatter(x, y, c=v, cmap="YlOrRd", s=1, vmin=0, vmax=5.0)
            ax2.set_title("Velocidad (m/s)")
            ax2.set_aspect("equal")
            plt.colorbar(sc2, ax=ax2)

            ruta = os.path.join(dir_fotos, f"paso_{int(t):05d}.png")
            plt.savefig(ruta, dpi=100)
            plt.close(fig)


# ================================================================
#  WORKER & ORCHESTRATOR
# ================================================================


def parse_args():
    p = argparse.ArgumentParser(description="ANUGA Flood Benchmark (Paiporta)")
    p.add_argument(
        "--resolucion",
        type=float,
        default=200.0,
        help="Resolución máxima de malla (m²)",
    )
    p.add_argument(
        "--duracion", type=float, default=3 * 3600, help="Duración de simulación (s)"
    )
    p.add_argument("--paso", type=float, default=60.0, help="Paso de guardado (s)")
    p.add_argument("--estrategia", choices=["elevar", "troquelar"], default="elevar")
    p.add_argument("--np", type=int, default=4, help="Número de procesos MPI")
    p.add_argument(
        "--run", type=str, default="cpu,mpi,gpu", help="Configuraciones: cpu,mpi,gpu"
    )
    p.add_argument("--worker", choices=["cpu", "mpi", "gpu"], help=argparse.SUPPRESS)
    return p.parse_args()


def run_worker(args):
    modo = args.worker
    is_mpi = modo == "mpi"
    modo_gpu = modo == "gpu"

    sim = SimulacionInundacion(
        esquina_so=[714619, 4365854],
        esquina_ne=[723413, 4370344],
        directorio=f"resultados/paiporta_{modo}/{args.resolucion}",
        nombre=f"dana_{modo}",
        ruta_mdt="PNOA_MDT05_ETRS89_HU30_0722_LID.tif",
        ruta_rugosidad="malla_rugosidad_qgis.gpkg",
        resolucion_malla=args.resolucion,
        estrategia_edificios=args.estrategia,
        altura_edificio=10.0,
        simplificar_tolerancia=1.0,
        area_minima_edificio=20.0,
        buffer_fusion=2.0,
        cache_edificios=True,
        generar_fotogramas=False,  # Disabled for benchmarking
    )

    def hidrograma_poyo(t):
        if t < 1800:
            return 5.0 + (2000.0 / 1800.0) * t
        elif t < 5400:
            return 1500.0 - (1500.0 / 3600.0) * (t - 1800)
        return 5.0

    def hidrograma_sur(t):
        if t < 2700:
            return 3.0 + (2000.0 / 2700.0) * t
        elif t < 7200:
            return 800.0 - (800.0 / 4500.0) * (t - 2700)
        return 3.0

    sim.agregar_inlet(
        centro=[715325, 4368947], caudal=hidrograma_poyo, nombre="Barranco Norte"
    )

    sim.agregar_inlet(
        centro=[715653.5, 4367070.7], caudal=hidrograma_sur, nombre="Barranco Sur"
    )

    wall_time, ntri = sim.ejecutar(
        duracion=args.duracion,
        paso_guardado=args.paso,
        forzar_malla=True,
        modo_gpu=modo_gpu,
    )

    if myid == 0:
        result = {
            "mode": modo,
            "wall_time": wall_time,
            "triangles": ntri,
            "nprocs": numprocs,
        }
        with open(f"/tmp/bench_{modo}_paiporta.json", "w") as f:
            json.dump(result, f)

    if is_mpi:
        finalize()


def run_benchmark(args):
    configs = [c.strip() for c in args.run.split(",") if c.strip()]
    valid = {"cpu", "mpi", "gpu"}
    for c in configs:
        if c not in valid:
            print(f"Unknown config '{c}'. Choose from: cpu, mpi, gpu")
            sys.exit(1)

    script = os.path.abspath(__file__)
    base_cmd = [
        sys.executable,
        script,
        "--resolucion",
        str(args.resolucion),
        "--duracion",
        str(args.duracion),
        "--paso",
        str(args.paso),
        "--estrategia",
        args.estrategia,
        "--np",
        str(args.np),
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    print("=" * 60)
    print(f"  ANUGA Flood Benchmark (Paiporta)")
    print(f"  Resolution: {args.resolucion} m²")
    print(f"  Simulation: {args.duracion:.0f}s, yieldstep {args.paso:.0f}s")
    print(f"  Running: {', '.join(configs)}")
    print("=" * 60)

    results = {}

    if "cpu" in configs:
        print("\n▶ Running: CPU (1 core) ...")
        r = subprocess.run(base_cmd + ["--worker", "cpu"], env=env)
        if r.returncode == 0 and os.path.exists("/tmp/bench_cpu_paiporta.json"):
            with open("/tmp/bench_cpu_paiporta.json") as f:
                results["cpu"] = json.load(f)

    if "mpi" in configs:
        print(f"\n▶ Running: MPI ({args.np} cores) ...")
        mpi_cmd = (
            ["mpiexec", "--allow-run-as-root", "--oversubscribe", "-np", str(args.np)]
            + base_cmd
            + ["--worker", "mpi"]
        )
        r = subprocess.run(mpi_cmd, env=env)
        if r.returncode == 0 and os.path.exists("/tmp/bench_mpi_paiporta.json"):
            with open("/tmp/bench_mpi_paiporta.json") as f:
                results["mpi"] = json.load(f)

    if "gpu" in configs:
        print("\n▶ Running: GPU ...")
        r = subprocess.run(base_cmd + ["--worker", "gpu"], env=env)
        if r.returncode == 0 and os.path.exists("/tmp/bench_gpu_paiporta.json"):
            with open("/tmp/bench_gpu_paiporta.json") as f:
                results["gpu"] = json.load(f)

    # ── Summary table ──
    if not results:
        print("\nNo successful runs.")
        return

    baseline_key = "cpu" if "cpu" in results else list(results.keys())[0]
    baseline_time = results[baseline_key]["wall_time"]

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("─" * 60)

    rows = [
        ("CPU (1 core)", "cpu"),
        (f"MPI ({args.np} cores)", "mpi"),
        ("GPU (mode=2)", "gpu"),
    ]

    fmt = "  {:<20s} {:>8s}   {:>8s}   {:>12s}"
    print(fmt.format("Config", "Time", "Speedup", "Triangles"))
    print(fmt.format("─" * 20, "─" * 8, "─" * 8, "─" * 12))

    for label, key in rows:
        if key not in configs:
            continue
        r = results.get(key)
        if r:
            t = r["wall_time"]
            tri = r.get("triangles", 0)
            nprocs = r.get("nprocs", 1)
            tri_label = f"{tri:,}"
            if key == "mpi":
                tri_label += f" ({tri * nprocs:,} tot)"
            sp = f"{baseline_time / t:.1f}x" if baseline_time else "—"
            ref = " (base)" if key == baseline_key else ""
            print(fmt.format(label, f"{t:.2f}s", sp + ref, tri_label))
        else:
            print(fmt.format(label, "FAILED", "—", "—"))

    print("─" * 60)


if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_benchmark(args)
