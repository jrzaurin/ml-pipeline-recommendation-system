mkdir data/raw/metadata_split
split -l 20000 data/raw/meta_Clothing_Shoes_and_Jewelry.json data/raw/metadata_split/part-

mkdir data/raw/reviews_split
split -l 200000 data/raw/Clothing_Shoes_and_Jewelry.json data/raw/reviews_split/part-
