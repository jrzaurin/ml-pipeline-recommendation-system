aws s3 cp data/raw/amazon/Movies_and_TV.json.gz s3://recsys-2/data/raw/amazon/Movies_and_TV.json.gz
aws s3 cp data/raw/amazon/meta_Movies_and_TV.json.gz s3://recsys-2/data/raw/amazon/meta_Movies_and_TV.json.gz

aws s3 cp s3://recsys-1/raw/Clothing_Shoes_and_Jewelry.json.gz data/raw/amazon/Clothing_Shoes_and_Jewelry.json.gz
aws s3 cp s3://recsys-1/raw/meta_Clothing_Shoes_and_Jewelry.json.gz data/raw/amazon/meta_Clothing_Shoes_and_Jewelry.json.gz

