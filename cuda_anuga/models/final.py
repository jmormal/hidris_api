#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clase unificada para simulaciones de inundación con ANUGA.
Versión optimizada — malla rápida + elevación corregida.

Cambios respecto a la versión original:
  1. Caché de edificios en disco (GeoPackage).
  2. Simplificación de geometrías (tolerance configurable).
  3. Fusión de edificios cercanos (buffer → unary_union → debuffer).
  4. Estrategia «elevar» (por defecto): no se usan agujeros en la malla;
     los edificios se modelan como celdas con elevación +50 m.
  5. Estrategia «troquelar» (la original): disponible si se necesita,
     pero con edificios simplificados y filtrados.
  6. Corrección geo_reference: ANUGA pasa coordenadas relativas al
     dominio, hay que sumar xllcorner/yllcorner antes de interpolar.
  7. Corrección interpolador: usa np.column_stack en vez de tupla.
"""

import os
import time
import logging
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
# Parche: Parallel_Inlet.statistics() — ver original
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
        # ── Parámetros de optimización ──
        estrategia_edificios: str = "elevar",
        altura_edificio: float = 50.0,
        simplificar_tolerancia: float = 1.0,
        area_minima_edificio: float = 20.0,
        buffer_fusion: float = 2.0,
        cache_edificios: bool = True,
    ):
        """
        Parámetros
        ----------
        esquina_so : [x_min, y_min]
            Esquina suroeste del rectángulo en coordenadas UTM.
        esquina_ne : [x_max, y_max]
            Esquina noreste del rectángulo en coordenadas UTM.
        directorio : str
            Carpeta donde se guardan (y buscan) todos los archivos.
        nombre : str
            Prefijo para los archivos generados (.msh, .sww, .gpkg…).
        epsg : int
            Código EPSG del sistema de referencia (por defecto ETRS89 / UTM 30N).
        resolucion_malla : float
            Área máxima de triángulo en m².
        ruta_mdt : str | None
            Ruta al GeoTIFF con el modelo digital del terreno.
        ruta_rugosidad : str | None
            Ruta al GPKG con la columna 'manning' por celda.
        manning_defecto : float
            Coeficiente de Manning uniforme si no hay GPKG de rugosidad.
        descargar_edificios : bool
            Si True, descarga edificios de OSM.
        generar_fotogramas : bool
            Si True, genera imágenes PNG en cada paso de guardado.
        estrategia_edificios : str  {"elevar", "troquelar"}
            - "elevar"   : malla simple + elevación alta en edificios (RÁPIDO).
            - "troquelar": agujeros en la malla (lento pero geométricamente exacto).
        altura_edificio : float
            Metros que se suman al MDT en las celdas de edificio (solo "elevar").
        simplificar_tolerancia : float
            Tolerancia en metros para simplificar polígonos de edificios.
        area_minima_edificio : float
            Área mínima en m² para conservar un edificio.
        buffer_fusion : float
            Radio en metros para fusionar edificios cercanos.
        cache_edificios : bool
            Si True, guarda los edificios descargados en un GPKG local.
        """
        self.log = logging.getLogger(f"Simulacion.{nombre}")

        # Geometría --------------------------------------------------
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

        # Rutas -------------------------------------------------------
        self.directorio = os.path.abspath(directorio)
        os.makedirs(self.directorio, exist_ok=True)
        self.nombre = nombre
        self.ruta_msh = os.path.join(self.directorio, f"{nombre}.msh")
        self.ruta_gpkg = os.path.join(self.directorio, f"{nombre}_malla.gpkg")
        self.ruta_cache_edificios = os.path.join(
            self.directorio, f"{nombre}_edificios.gpkg"
        )
        self.ruta_mdt = ruta_mdt
        self.ruta_rugosidad = ruta_rugosidad

        # Parámetros --------------------------------------------------
        self.resolucion_malla = resolucion_malla
        self.manning_defecto = manning_defecto
        self.descargar_edificios = descargar_edificios
        self.generar_fotogramas = generar_fotogramas

        # Optimización edificios --------------------------------------
        self.estrategia_edificios = estrategia_edificios
        self.altura_edificio = altura_edificio
        self.simplificar_tolerancia = simplificar_tolerancia
        self.area_minima_edificio = area_minima_edificio
        self.buffer_fusion = buffer_fusion
        self.cache_edificios = cache_edificios

        # Inlets (se añaden con .agregar_inlet()) --------------------
        self._inlets: list[dict] = []

        # Edificios procesados (se usan en _crear_dominio si
        # estrategia == "elevar")
        self._edificios_utm: gpd.GeoDataFrame | None = None

    # ================================================================
    #  API PÚBLICA
    # ================================================================

    def agregar_inlet(
        self,
        caudal,
        centro: list[float] | None = None,
        zona: list[list[float]] | None = None,
        lado: float | None = None,
        nombre: str = "",
    ):
        """
        Registra una fuente de agua.

        Se puede definir de dos formas (mutuamente excluyentes):
          - ``centro``: punto [x, y] UTM. El rectángulo se calcula
            automáticamente a partir de la resolución de malla.
          - ``zona``: polígono manual [[x,y], …] (uso avanzado).

        Parámetros
        ----------
        caudal : callable(t) → float  |  float
            Función hidrograma Q(t) en m³/s, o un caudal constante.
        centro : [x, y] | None
            Centro del inlet en coordenadas UTM.
        zona : lista de [x, y] | None
            Polígono UTM manual (alternativa a centro).
        lado : float | None
            Lado del cuadrado en metros. Si None se calcula como
            5 × √(resolución_malla).
        nombre : str
            Etiqueta descriptiva (solo para logging).
        """
        if centro is None and zona is None:
            raise ValueError("Debes indicar 'centro' o 'zona'.")
        if centro is not None and zona is not None:
            raise ValueError("Usa 'centro' o 'zona', no ambos.")

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
        self.log.info(f"💧 Inlet registrado: {self._inlets[-1]['nombre']}")

    def _calcular_zona_inlet(
        self, centro: list[float], lado: float | None
    ) -> list[list[float]]:
        """Genera un rectángulo centrado en ``centro``, recortado al dominio."""
        if lado is None:
            lado = 5.0 * np.sqrt(self.resolucion_malla)

        mitad = lado / 2.0
        cx, cy = centro

        if cx < self.x_min or cx > self.x_max or cy < self.y_min or cy > self.y_max:
            self.log.warning(f"⚠️  Centro ({cx}, {cy}) fuera del dominio. Se ajustará.")
            cx = np.clip(cx, self.x_min + mitad, self.x_max - mitad)
            cy = np.clip(cy, self.y_min + mitad, self.y_max - mitad)

        x0 = max(cx - mitad, self.x_min)
        x1 = min(cx + mitad, self.x_max)
        y0 = max(cy - mitad, self.y_min)
        y1 = min(cy + mitad, self.y_max)

        self.log.info(
            f"  📐 Inlet auto: centro=({cx:.0f}, {cy:.0f}), "
            f"recuadro={x1 - x0:.0f}×{y1 - y0:.0f} m"
        )
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    def ejecutar(
        self,
        duracion: float = 3 * 3600,
        paso_guardado: float = 60,
        forzar_malla: bool = False,
    ):
        """
        Lanza el pipeline completo.

        Parámetros
        ----------
        duracion : float
            Tiempo total de simulación en segundos.
        paso_guardado : float
            Intervalo entre fotogramas / yieldsteps en segundos.
        forzar_malla : bool
            Si True, regenera la malla aunque ya exista el .msh.
        """
        if not self._inlets:
            self.log.warning(
                "⚠️  No hay inlets registrados. Añade al menos uno con .agregar_inlet()."
            )

        # Paso 1 – Malla (solo proceso 0) ----------------------------
        if myid == 0:
            if os.path.exists(self.ruta_msh) and not forzar_malla:
                self.log.info(f"♻️  Malla existente: {self.ruta_msh} — se reutiliza.")
            else:
                self._generar_malla()

        barrier()

        # Paso 2 – Dominio (solo proceso 0) --------------------------
        domain = self._crear_dominio() if myid == 0 else None

        # Paso 3 – Distribuir -----------------------------------------
        self.log.info(f"📡 Proceso {myid}/{numprocs}: distribuyendo…")
        domain = distribute(domain)
        self.log.info(f"✅ Proceso {myid}: subdomain listo.")

        # Paso 4 – Inlets --------------------------------------------
        self._aplicar_inlets(domain)

        # Paso 5 – Simular -------------------------------------------
        self._simular(domain, duracion, paso_guardado)

        # Paso 6 – Fusionar ------------------------------------------
        barrier()
        domain.sww_merge()
        if myid == 0:
            self.log.info(f"💾 Resultado final: {self.nombre}.sww")
        finalize()

    # ================================================================
    #  EDIFICIOS — descarga, caché, simplificación, fusión
    # ================================================================

    def _obtener_edificios(self, poly_geom: Polygon) -> gpd.GeoDataFrame | None:
        """
        Obtiene edificios: desde caché si existe, si no desde OSM.
        Aplica simplificación, filtro de área y fusión.
        Devuelve un GeoDataFrame en UTM o None si no hay edificios.
        """
        # ── 1. Caché ────────────────────────────────────────────────
        if self.cache_edificios and os.path.exists(self.ruta_cache_edificios):
            self.log.info(f"♻️  Edificios desde caché: {self.ruta_cache_edificios}")
            edificios_utm = gpd.read_file(self.ruta_cache_edificios)
        else:
            # ── 2. Descarga OSM ─────────────────────────────────────
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
                # Conservar solo geometría
                edificios_utm = gpd.GeoDataFrame(
                    geometry=edificios_utm.geometry.values,
                    crs=f"EPSG:{self.epsg}",
                )
            except Exception as e:
                self.log.warning(f"⚠️ Error descargando edificios: {e}")
                return None

            # ── 3. Guardar caché ────────────────────────────────────
            if self.cache_edificios:
                edificios_utm.to_file(self.ruta_cache_edificios, driver="GPKG")
                self.log.info(
                    f"  💾 Caché guardada: {self.ruta_cache_edificios} "
                    f"({len(edificios_utm)} edificios)"
                )

        n_original = len(edificios_utm)
        v_original = sum(
            len(g.exterior.coords)
            if g.geom_type == "Polygon"
            else sum(len(p.exterior.coords) for p in g.geoms)
            for g in edificios_utm.geometry
        )

        # ── 4. Filtrar por área mínima ──────────────────────────────
        edificios_utm = edificios_utm[
            edificios_utm.geometry.area >= self.area_minima_edificio
        ]
        self.log.info(
            f"  🔍 Filtro área ≥ {self.area_minima_edificio} m²: "
            f"{n_original} → {len(edificios_utm)} edificios"
        )

        # ── 5. Simplificar geometrías ───────────────────────────────
        if self.simplificar_tolerancia > 0:
            edificios_utm["geometry"] = edificios_utm.geometry.simplify(
                self.simplificar_tolerancia, preserve_topology=True
            )
            edificios_utm = edificios_utm[
                edificios_utm.is_valid & ~edificios_utm.is_empty
            ]

        # ── 6. Fusionar edificios cercanos ──────────────────────────
        if self.buffer_fusion > 0:
            merged = unary_union(
                edificios_utm.geometry.buffer(self.buffer_fusion)
            ).buffer(-self.buffer_fusion)

            if merged.geom_type == "Polygon":
                polys = [merged]
            elif merged.geom_type == "MultiPolygon":
                polys = list(merged.geoms)
            else:
                polys = []

            edificios_utm = gpd.GeoDataFrame(geometry=polys, crs=f"EPSG:{self.epsg}")

        v_final = sum(
            len(g.exterior.coords)
            if g.geom_type == "Polygon"
            else sum(len(p.exterior.coords) for p in g.geoms)
            for g in edificios_utm.geometry
        )
        self.log.info(
            f"  ✅ Resultado: {len(edificios_utm)} polígonos, "
            f"{v_original} → {v_final} vértices "
            f"({100 * (1 - v_final / max(v_original, 1)):.0f} % reducción)"
        )

        return edificios_utm

    # ================================================================
    #  MALLA
    # ================================================================

    def _generar_malla(self):
        """Genera el .msh, usando agujeros solo si estrategia='troquelar'."""
        self.log.info("🚀 Generando malla…")
        t0 = time.time()

        poly_geom = Polygon(self.poligono_utm)
        agujeros = []

        if self.descargar_edificios:
            edificios = self._obtener_edificios(poly_geom)

            if edificios is not None and len(edificios) > 0:
                if self.estrategia_edificios == "troquelar":
                    agujeros = self._gdf_a_agujeros(edificios)
                else:
                    self._edificios_utm = edificios
                    self.log.info(
                        "  🏗️ Estrategia «elevar»: malla sin agujeros, "
                        "edificios se aplican como elevación."
                    )

        boundary_tags = {"sur": [0], "este": [1], "norte": [2], "oeste": [3]}

        anuga.create_pmesh_from_regions(
            self.poligono_utm,
            boundary_tags=boundary_tags,
            maximum_triangle_area=self.resolucion_malla,
            interior_holes=agujeros,
            filename=self.ruta_msh,
        )

        dt = time.time() - t0
        self.log.info(f"✅ Malla guardada: {self.ruta_msh}  ({dt:.1f} s)")
        self._exportar_gpkg()

    @staticmethod
    def _gdf_a_agujeros(gdf: gpd.GeoDataFrame) -> list:
        """Convierte un GeoDataFrame a lista de coordenadas (para interior_holes)."""
        agujeros = []
        for geom in gdf.geometry:
            if geom.geom_type == "Polygon":
                agujeros.append(list(geom.exterior.coords))
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    agujeros.append(list(poly.exterior.coords))
        return agujeros

    def _exportar_gpkg(self):
        """Exporta la malla como GeoPackage para QGIS."""
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
            self.log.info(f"✅ GPKG visual: {self.ruta_gpkg}")
        except Exception as e:
            self.log.warning(f"⚠️ No se pudo exportar GPKG: {e}")

    # ================================================================
    #  DOMINIO
    # ================================================================

    def _crear_dominio(self) -> anuga.Domain:
        """Configura el dominio ANUGA: elevación, rugosidad y contornos."""
        self.log.info("📂 Creando dominio desde malla…")
        domain = anuga.Domain(self.ruta_msh)
        domain.set_name(self.nombre)
        domain.set_datadir(self.directorio)
        domain.set_default_order(2)
        domain.set_minimum_storable_height(0.01)

        # Rugosidad ---------------------------------------------------
        if self.ruta_rugosidad and os.path.exists(self.ruta_rugosidad):
            self.log.info("🌿 Cargando rugosidad desde GPKG…")
            try:
                gdf = gpd.read_file(self.ruta_rugosidad)
                domain.set_quantity("friction", gdf["manning"].values)
            except Exception as e:
                self.log.warning(f"⚠️ Rugosidad fallback a {self.manning_defecto}: {e}")
                domain.set_quantity("friction", self.manning_defecto)
        else:
            domain.set_quantity("friction", self.manning_defecto)

        # Cargar edificios si no están en memoria (p.ej. malla reutilizada)
        if (
            self.estrategia_edificios == "elevar"
            and self.descargar_edificios
            and self._edificios_utm is None
        ):
            poly_geom = Polygon(self.poligono_utm)
            self._edificios_utm = self._obtener_edificios(poly_geom)

        # Elevación ---------------------------------------------------
        if self.ruta_mdt and os.path.exists(self.ruta_mdt):
            self.log.info("🏔️ Proyectando MDT sobre la malla…")
            interp = self._crear_interpolador_mdt()

            # ANUGA pasa coordenadas RELATIVAS al geo_reference,
            # hay que sumar xllcorner / yllcorner para obtener UTM absoluto.
            geo = domain.geo_reference
            xll = geo.get_xllcorner()
            yll = geo.get_yllcorner()
            self.log.info(f"  📍 geo_reference: xll={xll:.1f}, yll={yll:.1f}")

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
                    "elevation",
                    elevacion_con_edificios,
                    location="centroids",
                )
                self.log.info(
                    f"  🏗️ Elevación +{self.altura_edificio} m aplicada "
                    f"a celdas dentro de {len(self._edificios_utm)} polígonos."
                )
            else:
                domain.set_quantity(
                    "elevation",
                    lambda x, y: interp(np.column_stack([y + yll, x + xll])),
                    location="centroids",
                )

            # Verificar elevación
            elev_values = domain.get_quantity("elevation").get_values(
                location="centroids"
            )
            self.log.info(
                f"  📊 Elevación: min={elev_values.min():.2f} m, "
                f"max={elev_values.max():.2f} m, "
                f"mean={elev_values.mean():.2f} m"
            )
        else:
            self.log.warning("⚠️ Sin MDT — elevación a 0.")
            domain.set_quantity("elevation", 0.0)

        # Condición inicial: terreno seco
        domain.set_quantity("stage", expression="elevation")

        # Contornos ---------------------------------------------------
        self.log.info("🧱 Configurando condiciones de contorno…")
        Br = anuga.Reflective_boundary(domain)
        Bt = anuga.Transmissive_boundary(domain)
        tags = {"sur": Bt, "este": Bt, "norte": Br, "oeste": Br}
        if "interior" in domain.get_boundary_tags():
            tags["interior"] = Br
        domain.set_boundary(tags)

        return domain

    def _crear_interpolador_mdt(self) -> RegularGridInterpolator:
        """Lee el GeoTIFF del MDT y devuelve un interpolador 2D."""
        with rasterio.open(self.ruta_mdt) as src:
            elev = src.read(1).astype(np.float64)
            elev[elev < -999] = np.nan
            x = np.linspace(src.bounds.left, src.bounds.right, src.width)
            y = np.linspace(src.bounds.top, src.bounds.bottom, src.height)
            return RegularGridInterpolator(
                (y[::-1], x),
                np.flipud(elev),
                bounds_error=False,
                fill_value=0,
            )

    def _aplicar_inlets(self, domain):
        """Registra todos los inlets sobre el dominio (ya distribuido)."""
        for inlet in self._inlets:
            self.log.info(f"💧 Activando inlet: {inlet['nombre']}")
            anuga.Inlet_operator(domain, inlet["zona"], Q=inlet["caudal"])

    def _simular(self, domain, duracion: float, paso: float):
        """Bucle principal de evolución temporal."""
        if self.generar_fotogramas:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

        dir_fotos = os.path.join(self.directorio, "fotogramas")
        if myid == 0 and self.generar_fotogramas:
            os.makedirs(dir_fotos, exist_ok=True)
        barrier()

        t0 = time.time()

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

            self.log.info(
                f"🌊 t={t / 60:6.1f} min | calado_max={np.max(h):.2f} m "
                f"| Δreal={time.time() - t0:.0f}s"
            )

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
#  EJEMPLO DE US
# ================================================================
if __name__ == "__main__":
    resolucion_malla = 200
    sim = SimulacionInundacion(
        esquina_so=[714619, 4365854],
        esquina_ne=[723729, 4369389],
        directorio=f"resultados/paiporta/{resolucion_malla}",
        nombre="dana_paiporta",
        ruta_mdt="PNOA_MDT05_ETRS89_HU30_0722_LID.tif",
        ruta_rugosidad="malla_rugosidad_qgis.gpkg",
        resolucion_malla=150,
        # ── Opciones de optimización ──
        estrategia_edificios="elevar",
        altura_edificio=50.0,
        simplificar_tolerancia=1.0,
        area_minima_edificio=20.0,
        buffer_fusion=2.0,
        cache_edificios=True,
    )

    def hidrograma_poyo(t):
        """Pico de 1500 m³/s a los 30 min, descenso en 1h."""
        if t < 1800:
            return 5.0 + (1500.0 / 1800.0) * t
        elif t < 5400:
            return 1500.0 - (1500.0 / 3600.0) * (t - 1800)
        return 5.0

    def hidrograma_sur(t):
        """Pico secundario de 800 m³/s a los 45 min."""
        if t < 2700:
            return 3.0 + (800.0 / 2700.0) * t
        elif t < 7200:
            return 800.0 - (800.0 / 4500.0) * (t - 2700)
        return 3.0

    sim.agregar_inlet(
        centro=[714168.4, 4370133.9],
        caudal=hidrograma_poyo,
        nombre="Barranco Norte",
    )

    sim.agregar_inlet(
        centro=[715653.5, 4367070.7],
        caudal=hidrograma_sur,
        nombre="Barranco Sur",
    )

    sim.ejecutar(duracion=60, paso_guardado=60, forzar_malla=True)
