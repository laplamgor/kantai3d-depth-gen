"""Build the server paths and local file names for ship CG assets.

Only ship full-res CG paths are produced here (full + full_dmg). The naming
scheme mirrors how the game client requests its sprites:

    <sortno>_<hash>_<filename>.png?version=<ver>

where ``hash`` is a deterministic function of the ship id and the resource
kind. The ``_v<ver>`` suffix on the *local* file name is our own
disambiguation so a re-rendered CG does not collide with the previous one.

Verified against files that already exist in our `input` folder, e.g. ship 1
yields 0001_5865_snohitatusbk_v54.png and 0001_6245_snohitatusbk_v54.png.
"""

# Deterministic hash table used to derive the per-resource path component.
RESOURCE = [
    6657, 5699, 3371, 8909, 7719, 6229, 5449, 8561, 2987, 5501, 3127, 9319,
    4365, 9811, 9927, 2423, 3439, 1865, 5925, 4409, 5509, 1517, 9695, 9255,
    5325, 3691, 5519, 6949, 5607, 9539, 4133, 7795, 5465, 2659, 6381, 6875,
    4019, 9195, 5645, 2887, 1213, 1815, 8671, 3015, 3147, 2991, 7977, 7045,
    1619, 7909, 4451, 6573, 4545, 8251, 5983, 2849, 7249, 7449, 9477, 5963,
    2711, 9019, 7375, 2201, 5631, 4893, 7653, 3719, 8819, 5839, 1853, 9843,
    9119, 7023, 5681, 2345, 9873, 6349, 9315, 3795, 9737, 4633, 4173, 7549,
    7171, 6147, 4723, 5039, 2723, 7815, 6201, 5999, 5339, 4431, 2911, 4435,
    3611, 4423, 9517, 3243,
]

# Ships to skip (no usable CG / not wanted).
IGNORED_IDS = {3, 4, 5, 8, 315, 333, 335, 336, 337, 338, 339, 340, 341, 342, 482}


def _key(s):
    """Sum of char codes."""
    return sum(ord(c) for c in s)


def _create(ship_id, kind):
    """Deterministic hash suffix for a (ship, resource kind) pair."""
    n = len(kind)
    return (17 * (ship_id + 7) * RESOURCE[(_key(kind) + ship_id * n) % 100] % 8973 + 1000)


def _pad(ship_id):
    return str(ship_id).zfill(4)


def ship_cg_basename(ship_id, filename, dmg=False):
    """Return the server-side base file name (no version suffix).

    This is the name the KC server actually serves, e.g.
    ``0001_5865_snohitatusbk.png``. The ``?version=`` query is appended
    separately by :func:`ship_cg_entries`.
    """
    kind = "ship_full_dmg" if dmg else "ship_full"
    tail = f"{_pad(ship_id)}_{_create(ship_id, kind)}"
    if filename:
        tail += f"_{filename}"
    return f"{tail}.png"


def ship_cg_filename(ship_id, filename, version, dmg=False):
    """Return the *local* CG file name for one ship image.

      0001_5865_snohitatusbk.png?version=54  ->  0001_5865_snohitatusbk_v54.png

    When the version is empty or "1" no version suffix is added (version "1"
    means the base file).
    """
    base = ship_cg_basename(ship_id, filename, dmg=dmg)
    ver = "" if version in (None, "", "1") else str(version)
    if ver:
        return base[:-4] + f"_v{ver}.png"  # base[:-4] drops ".png"
    return base


def ship_cg_entries(start2):
    """Yield (filename, remote_path) for every ship full CG in start2.

    ``remote_path`` is the server-relative URL path shipped to the KC server,
    e.g. ``kcs2/resources/ship/full/0001_5865_snohitatusbk.png?version=54``.
    """
    seen = set()
    for ship in start2.get("api_mst_shipgraph", []):
        if ship.get("api_sortno") == 0 and ship.get("api_boko_d"):
            continue  # friendly unused
        api_id = ship.get("api_id")
        if api_id is None or api_id >= 1500 or api_id in IGNORED_IDS:
            continue
        filename = ship.get("api_filename", "")
        versions = ship.get("api_version") or []
        version = versions[0] if versions else ""

        for dmg in (False, True):
            local = ship_cg_filename(api_id, filename, version, dmg=dmg)
            if local in seen:
                continue
            seen.add(local)
            folder = "full_dmg" if dmg else "full"
            # The server serves the UN-versioned base name; the version is
            # only a ?version= cache-buster query (and the _v<ver> suffix is
            # purely a local disambiguation of the saved file).
            base = ship_cg_basename(api_id, filename, dmg=dmg)
            remote = f"kcs2/resources/ship/{folder}/{base}"
            if version and version != "1":
                remote += f"?version={version}"
            yield local, remote
