python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
gdown "1WgWoyB8AneE-4s0-6Lnh0KuUa6Wz9ra-"
mv seq2seq_vanilla.zip src/
cd src/
unzip seq2seq_vanilla.zip
rm seq2seq_vanilla.zip
