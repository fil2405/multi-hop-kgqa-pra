#type: ignore
HOP = 1 #hop: 1, 2, 3

TRAIN_PATH = f"dataset_processed/{HOP}-hop/train_set.parquet"
DEV_PATH = f"dataset_processed/{HOP}-hop/dev_set.parquet"
TEST_PATH = f"dataset_processed/{HOP}-hop/test_set.parquet"

REL_KEYWORDS = {
    'directed_by': ['direct'],
    'starred_actors': ['star', 'act', 'cast', 'play'],
    'written_by': ['writ', 'author', 'screenplay', 'script'],
    'release_year': ['year', 'release', 'when'],
    'has_genre': ['genre', 'kind', 'type'],
    'in_language': ['language', 'speak', 'spoken'],
    'has_tags': ['tag', 'about', 'subject', 'topic'],
    'has_imdb_rating': ['rating', 'score', 'rate'],
    'has_imdb_votes': ['vote', 'votes'],
}