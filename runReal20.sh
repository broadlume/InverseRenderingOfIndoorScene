# TEST
python3 -m cProfile -s cumtime testReal.py --cuda --dataRoot ../datasets/test_images_png --imList imList_20.txt \
    --testRoot Real20 --isLight --level 2 \
    --experiment0 models/check_cascade0_w320_h240 --nepoch0 14 \
    --experimentLight0 models/check_cascadeLight0_sg12_offset1 --nepochLight0 10 \
    --experiment1 models/check_cascade1_w320_h240 --nepoch1 7 \
    --experimentLight1 models/check_cascadeLight1_sg12_offset1 --nepochLight1 10 > cprofile.txt
