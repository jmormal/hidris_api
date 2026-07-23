-- ============================================================
-- Martin-published functions (CORRECTED SIGNATURES).
--
-- Martin auto-detects functions with the signature:
--     (z integer, x integer, y integer, query_params json) RETURNS bytea
-- The first three args are the tile coordinates; the OPTIONAL fourth arg
-- must be a single `json` parameter. Martin maps the URL query string into
-- that json object. Extra typed scalar args (storm_id integer, recorded_at
-- text, ...) are NOT recognized — a function with that shape is silently
-- skipped by auto-discovery and cannot be bound by an explicit `functions:`
-- mapping either. So every tile param is pulled out of query_params here.
--
-- Endpoints once Martin is running:
--   /storm_tile/{z}/{x}/{y}?storm_id=1&recorded_at=2026-06-30T14:30:00Z
--   /storm_tile/{z}/{x}/{y}?storm_id=1&recorded_at=...&measure_type=rain_rate
--   /storm_accum6h/{z}/{x}/{y}?storm_id=1&end_at=2026-06-30T14:00:00Z
--   value: SELECT storm_value(1,'2026-06-30T14:30:00Z',-0.37,39.47);
--
-- NOTE ON CONTENT TYPE: these functions return PNG (ST_AsPNG), i.e. RASTER
-- tiles, not MVT vector tiles. Martin advertises function sources as
-- application/x-protobuf by default. To serve PNG correctly, override the
-- TileJSON via an SQL comment on the function (see the COMMENT blocks at the
-- bottom), and consume them as a raster/bitmap layer on the client, not a
-- vector layer.
--
-- Color ranges:
--   rain_depth  per-frame mm: 0..50
--   rain_rate   mm/hr:        0..50
--   accum_6h    mm total:     0..150
-- ============================================================

-- ------------------------------------------------------------
-- Drop every prior overload so the tile name is unambiguous to Martin.
-- Includes both the old timestamptz signatures and the scalar-arg
-- signatures that did not auto-discover.
-- ------------------------------------------------------------
DROP FUNCTION IF EXISTS storm_tile(integer, integer, integer, integer, timestamptz, text);
DROP FUNCTION IF EXISTS storm_tile(integer, integer, integer, integer, text, text);
DROP FUNCTION IF EXISTS storm_accum6h(integer, integer, integer, integer, timestamptz, text);
DROP FUNCTION IF EXISTS storm_accum6h(integer, integer, integer, integer, text, text);
DROP FUNCTION IF EXISTS storm_value(integer, timestamptz, double precision, double precision, text);
DROP FUNCTION IF EXISTS storm_value_accum6h(integer, timestamptz, double precision, double precision, text);

-- ------------------------------------------------------------
-- Shared colormap helper: explicit value->color ramp (ST_ColorMap format).
-- The band is rescaled into 0..255 8BUI first so one ramp works for any
-- physical range.
-- ------------------------------------------------------------
CREATE OR REPLACE FUNCTION _ramp_for(measure_type text)
RETURNS text
LANGUAGE sql IMMUTABLE AS $$
  -- 8-bit ramp (after rescale): transparent at 0, blue->green->yellow->red.
  SELECT $ramp$
0   0   0   0   0
1   0   0   255 180
64  0   200 255 220
128 0   220 60  230
192 255 220 0   240
255 200 0   0   255
$ramp$;
$$;

-- physical max used to rescale each product into 0..255
CREATE OR REPLACE FUNCTION _phys_max(measure_type text)
RETURNS double precision
LANGUAGE sql IMMUTABLE AS $$
  SELECT CASE measure_type
           WHEN 'rain_depth' THEN 50.0    -- mm per frame
           WHEN 'rain_rate'  THEN 50.0    -- mm/hr
           WHEN 'accum_6h'   THEN 150.0   -- mm total over 6h
           ELSE 50.0
         END;
$$;

-- ------------------------------------------------------------
-- Instantaneous tile for one frame.
-- Martin signature: (z, x, y, query_params json) RETURNS bytea.
-- Params read from query_params:
--   storm_id     (required)
--   recorded_at  (required, ISO timestamp, cast to timestamptz)
--   measure_type (optional, default 'rain_depth'; pass 'rain_rate' to override)
-- ------------------------------------------------------------
CREATE OR REPLACE FUNCTION storm_tile(
    z integer, x integer, y integer,
    query_params json
)
RETURNS bytea
LANGUAGE plpgsql STABLE PARALLEL SAFE AS $$
DECLARE
    in_storm_id     integer     := (query_params->>'storm_id')::integer;
    rec_ts          timestamptz := (query_params->>'recorded_at')::timestamptz;
    in_measure_type text        := COALESCE(query_params->>'measure_type', 'rain_depth');
    env     geometry := ST_TileEnvelope(z, x, y);       -- 3857
    env4326 geometry := ST_Transform(env, 4326);
    merged  raster;
    png     bytea;
    pmax    double precision := _phys_max(in_measure_type);
BEGIN
    PERFORM set_config('postgis.gdal_enabled_drivers', 'ENABLE_ALL', true);

    SELECT ST_Union(rast) INTO merged
    FROM storm_raster_data d
    WHERE d.storm_id     = in_storm_id
      AND d.measure_type = in_measure_type
      AND d.recorded_at  = rec_ts
      AND ST_Intersects(d.rast, env4326);

    IF merged IS NULL THEN
        RETURN NULL;   -- empty tile -> Martin returns 204
    END IF;

    -- reproject to web mercator, clip to the tile, rescale to 0..255, colorize
    merged := ST_Transform(merged, 3857);
    merged := ST_Clip(merged, env, true);
    -- rescale physical 0..pmax into 0..255 8BUI (values above pmax clamp)
    merged := ST_Reclass(merged, 1,
                 format('[0-%s):0-255, [%s-100000]:255', pmax, pmax),
                '8BUI', 0);
    png := ST_AsPNG(ST_ColorMap(merged, 1, _ramp_for(in_measure_type)));
    RETURN png;
END;
$$;

-- ------------------------------------------------------------
-- 6-hour accumulation tile, ending at end_at (trailing window).
-- Martin signature: (z, x, y, query_params json) RETURNS bytea.
-- Params read from query_params:
--   storm_id     (required)
--   end_at       (required, ISO timestamp, cast to timestamptz)
--   measure_type (optional, default 'rain_depth'; 'rain_rate' weights by interval)
-- For rain_depth: straight SUM of frames in (end_at-6h, end_at].
-- For rain_rate : weight each frame by interval_min/60 before summing.
-- ------------------------------------------------------------
CREATE OR REPLACE FUNCTION storm_accum6h(
    z integer, x integer, y integer,
    query_params json
)
RETURNS bytea
LANGUAGE plpgsql STABLE PARALLEL SAFE AS $$
DECLARE
    in_storm_id     integer     := (query_params->>'storm_id')::integer;
    end_ts          timestamptz := (query_params->>'end_at')::timestamptz;
    in_measure_type text        := COALESCE(query_params->>'measure_type', 'rain_depth');
    env      geometry := ST_TileEnvelope(z, x, y);
    env4326  geometry := ST_Transform(env, 4326);
    start_at timestamptz := end_ts - INTERVAL '6 hours';
    accum    raster;
    png      bytea;
    pmax     double precision := _phys_max('accum_6h');
BEGIN
    PERFORM set_config('postgis.gdal_enabled_drivers', 'ENABLE_ALL', true);

    IF in_measure_type = 'rain_rate' THEN
        -- depth_i = rate_i * interval_hours ; sum those
        SELECT ST_Union(
                 ST_MapAlgebra(rast, 1, NULL,
                   '[rast] * ' || (interval_min::double precision/60.0)::text),
                 'SUM')
          INTO accum
        FROM storm_raster_data d
        WHERE d.storm_id     = in_storm_id
          AND d.measure_type = 'rain_rate'
          AND d.recorded_at  > start_at AND d.recorded_at <= end_ts
          AND ST_Intersects(d.rast, env4326);
    ELSE
        -- rain_depth: already per-interval mm, just sum
        SELECT ST_Union(rast, 'SUM') INTO accum
        FROM storm_raster_data d
        WHERE d.storm_id     = in_storm_id
          AND d.measure_type = 'rain_depth'
          AND d.recorded_at  > start_at AND d.recorded_at <= end_ts
          AND ST_Intersects(d.rast, env4326);
    END IF;

    IF accum IS NULL THEN
        RETURN NULL;
    END IF;

    accum := ST_Transform(accum, 3857);
    accum := ST_Clip(accum, env, true);
    accum := ST_Reclass(accum, 1,
                format('[0-%s):0-255, [%s-100000]:255', pmax, pmax),
                '8BUI', 0);
    png := ST_AsPNG(ST_ColorMap(accum, 1, _ramp_for('accum_6h')));
    RETURN png;
END;
$$;

-- ------------------------------------------------------------
-- Point value: true stored value at a lon/lat for one frame.
-- NOT a tile function — call directly. Signature unchanged.
-- Returns the physical value (mm or mm/hr), not a color.
-- ------------------------------------------------------------
CREATE OR REPLACE FUNCTION storm_value(
    storm_id integer,
    recorded_at text,
    lon double precision,
    lat double precision,
    measure_type text DEFAULT 'rain_depth'
)
RETURNS double precision
LANGUAGE plpgsql STABLE PARALLEL SAFE AS $$
DECLARE
    rec_ts timestamptz := recorded_at::timestamptz;
    pt     geometry := ST_SetSRID(ST_Point(lon, lat), 4326);
    val    double precision;
BEGIN
    SELECT ST_Value(rast, pt) INTO val
    FROM storm_raster_data d
    WHERE d.storm_id = storm_value.storm_id
      AND d.measure_type = storm_value.measure_type
      AND d.recorded_at = rec_ts
      AND ST_Intersects(d.rast, pt)
    LIMIT 1;
    RETURN val;
END;
$$;

-- 6h-accumulated point value (trailing window ending at end_at)
CREATE OR REPLACE FUNCTION storm_value_accum6h(
    storm_id integer,
    end_at text,
    lon double precision,
    lat double precision,
    measure_type text DEFAULT 'rain_depth'
)
RETURNS double precision
LANGUAGE plpgsql STABLE PARALLEL SAFE AS $$
DECLARE
    end_ts   timestamptz := end_at::timestamptz;
    pt       geometry := ST_SetSRID(ST_Point(lon, lat), 4326);
    start_at timestamptz := end_ts - INTERVAL '6 hours';
    total    double precision;
BEGIN
    IF measure_type = 'rain_rate' THEN
        SELECT sum(ST_Value(rast, pt) * (interval_min::double precision/60.0))
          INTO total
        FROM storm_raster_data d
        WHERE d.storm_id = storm_value_accum6h.storm_id
          AND d.measure_type = 'rain_rate'
          AND d.recorded_at > start_at AND d.recorded_at <= end_ts
          AND ST_Intersects(d.rast, pt);
    ELSE
        SELECT sum(ST_Value(rast, pt)) INTO total
        FROM storm_raster_data d
        WHERE d.storm_id = storm_value_accum6h.storm_id
          AND d.measure_type = 'rain_depth'
          AND d.recorded_at > start_at AND d.recorded_at <= end_ts
          AND ST_Intersects(d.rast, pt);
    END IF;
    RETURN total;
END;
$$;

-- ------------------------------------------------------------
-- Grants for the Martin connection user (hidris).
-- ------------------------------------------------------------
GRANT USAGE ON SCHEMA public TO hidris;
GRANT EXECUTE ON FUNCTION storm_tile(integer, integer, integer, json) TO hidris;
GRANT EXECUTE ON FUNCTION storm_accum6h(integer, integer, integer, json) TO hidris;
GRANT EXECUTE ON FUNCTION _ramp_for(text), _phys_max(text) TO hidris;
GRANT EXECUTE ON FUNCTION
  storm_value(integer, text, double precision, double precision, text),
  storm_value_accum6h(integer, text, double precision, double precision, text)
TO hidris;

-- ------------------------------------------------------------
-- TileJSON content-type override so Martin serves PNG, not protobuf.
-- Martin merges a valid-JSON SQL comment into the generated TileJSON via
-- JSON Merge Patch. EXECUTE format guarantees the comment is valid JSON.
-- ------------------------------------------------------------
COMMENT ON FUNCTION storm_tile(integer, integer, integer, json) IS
  '{"content_type":"image/png"}';
COMMENT ON FUNCTION storm_accum6h(integer, integer, integer, json) IS
  '{"content_type":"image/png"}';
