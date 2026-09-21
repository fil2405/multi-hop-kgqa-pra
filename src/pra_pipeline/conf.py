#type: ignore
from pathlib import Path
import os

HOP = 3

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_processed" / f"{HOP}-hop"

TRAIN_PATH = DATASET_DIR / "train_set.parquet"
DEV_PATH = DATASET_DIR / "dev_set.parquet"
TEST_PATH = DATASET_DIR / "test_set.parquet"

REL_KEYWORDS = {
    'directed_by': ['direct'],
    'starred_actors': ['star', 'act', 'cast', 'play'],
    'written_by': ['writ', 'wrote', 'author', 'screenplay', 'script'],
    'release_year': ['year', 'release', 'when'],
    'has_genre': ['genre', 'kind', 'type'],
    'in_language': ['language', 'speak', 'spoken'],
    'has_tags': ['tag', 'about', 'subject', 'topic'],
    'has_imdb_rating': ['rating', 'score', 'rate'],
    'has_imdb_votes': ['vote', 'votes'],
}