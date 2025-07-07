from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from tinytag import TinyTag, UnsupportedFormatError
from tqdm import tqdm

music_dir = Path("/Users/user/Main/Music")
xl_path = Path("/Users/user/Main/Repos/spotify-scripts/liked_tracks.xlsx")
df = pd.read_excel(xl_path)
artist_to_track_albums: dict[str, list[tuple[str, str]]] = defaultdict(list)
cur_artist: str | None = None
for _, row in df.iterrows():
    if isinstance(row["artists"], str):
        cur_artist = row["artists"].lower()

    if cur_artist is not None:
        artist_to_track_albums[cur_artist].append((row["name"].lower(), row["album"].lower()))

for p in tqdm(list(music_dir.rglob("*"))):
    if p.is_dir():
        continue
    try:
        track = TinyTag.get(str(p))
    except UnsupportedFormatError:
        continue
    title = track.title
    artist = track.artist
    if artist is None or title is None:
        continue

    title, artist = title.lower(), artist.lower()

    for cur_artist, cur_tracks in artist_to_track_albums.items():
        if fuzz.partial_ratio(cur_artist, artist) < 90:
            continue
        for cur_title, cur_album in cur_tracks.copy():
            if fuzz.partial_ratio(cur_title, title) > 90:
                cur_tracks.remove((cur_title, cur_album))

rows: list[dict[str, str]] = []
for artist, tracks in artist_to_track_albums.items():
    for title, album in tracks:
        rows.append({"artist": artist, "album": album, "title": title})

df = pd.DataFrame(rows).set_index(["artist", "album"])
df.to_excel(xl_path.with_stem("liked_tracks_pruned"))
