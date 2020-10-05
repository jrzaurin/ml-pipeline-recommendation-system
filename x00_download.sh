mkdir -p data
mkdir -p data/raw
cd data/raw
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Clothing_Shoes_and_Jewelry.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Clothing_Shoes_and_Jewelry.csv
wget http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Clothing_Shoes_and_Jewelry.json.gz
wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Clothing_Shoes_and_Jewelry.json.gz


gunzip meta_Clothing_Shoes_and_Jewelry.json.gz
gunzip Clothing_Shoes_and_Jewelry.json.gz
