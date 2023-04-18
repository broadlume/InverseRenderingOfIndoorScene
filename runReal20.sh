# TEST
python3 -m cProfile -s cumtime testReal.py --cuda --dataRoot ../datasets/test_images_png --imList imList_20.txt \
    --testRoot Real20 --isLight --level 2 > cprofile.txt
