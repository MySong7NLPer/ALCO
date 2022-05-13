#rm -r $SMYDATA/fast/extractor_test_ext2

#CUDA_VISIBLE_DEVICES=6 python train_extractor_ml.py --path=$SMYDATA/fast/extractor_test_ext2/model --w2v=$SMYDATA/fast/word2vec/word2vec.128d.226k.bin

#rm -r $SMYDATA/fast/abstractor_dual

#CUDA_VISIBLE_DEVICES=0 python train_abstractor.py --path=$SMYDATA/fast/abstractor_dual/model --w2v=$SMYDATA/fast/word2vec/word2vec.128d.226k.bin

#rm -r $SMYDATA/bigmodel/abstractor_dual

#CUDA_VISIBLE_DEVICES=3 python train_abstractor.py --path=$SMYDATA/bigmodel/abstractor_dual/model --w2v=$SMYDATA/bigPatentData/word2vec/word2vec.128d.816k.bin


#rm -r $SMYDATA/fast/test_sem/decoded
#python decode_full_model.py --no-cuda --path=$SMYDATA/fast/test_sem/decoded/ --model_dir=$SMYDATA/fast/pretrained/model --beam=5 --test

rm -r $SMYDATA/fast/save_alco_001

CUDA_VISIBLE_DEVICES=3 python train_full_rl.py --path=$SMYDATA/fast/save_alco_001/model --abs_dir=$SMYDATA/fast/abstractor_more_baseline/model --ext_dir=$SMYDATA/fast/extractor_baseline/model

rm -r $SMYDATA/fast/save_alco_001/decoded

CUDA_VISIBLE_DEVICES=3 python decode_full_model.py --path=$SMYDATA/fast/save_alco_001/decoded/ --model_dir=$SMYDATA/fast/save_alco_001/model --beam=5 --test

#rm -r $SMYDATA/fast/save_dual_abs_newdata_comb8

#CUDA_VISIBLE_DEVICES=2 python train_full_rl.py --path=$SMYDATA/fast/save_dual_abs_newdata_comb8/model --abs_dir=$SMYDATA/fast/abstractor_more_baseline/model --ext_dir=$SMYDATA/fast/extractor_baseline/model

#rm -r $SMYDATA/fast/save_dual_abs_newdata_comb8/decoded

#CUDA_VISIBLE_DEVICES=2 python decode_full_model.py --path=$SMYDATA/fast/save_dual_abs_newdata_comb8/decoded/ --model_dir=$SMYDATA/fast/save_dual_abs_newdata_comb8/model --beam=5 --test

#rm -r $SMYDATA/bigmodel/comb_b_1

#CUDA_VISIBLE_DEVICES=3 python train_full_rl.py --path=$SMYDATA/bigmodel/comb_b_1/model --abs_dir=$SMYDATA/bigmodel/abstractor/model --ext_dir=$SMYDATA/bigmodel/extractor/model

#rm -r $SMYDATA/bigmodel/comb_b_1/decoded

#CUDA_VISIBLE_DEVICES=3 python decode_full_model.py --path=$SMYDATA/bigmodel/comb_b_1/decoded/ --model_dir=$SMYDATA/bigmodel/comb_b_1/model --beam=5 --test


#rm -r $SMYDATA/fast/save_dual_abs_newdata_1

#CUDA_VISIBLE_DEVICES=0 python train_full_rl.py --path=$SMYDATA/fast/save_dual_abs_newdata_1/model --abs_dir=$SMYDATA/fast/abstractor_more_baseline/model --ext_dir=$SMYDATA/fast/extractor_baseline/model

#rm -r $SMYDATA/fast/save_dual_abs_newdata_1/decoded

#python decode_full_model.py --path=$SMYDATA/fast/save_dual_abs_newdata_1/decoded/ --model_dir=$SMYDATA/fast/save_dual_abs_newdata_1/model --beam=5 --test



#--------------------
#test

#rm -r $SMYDATA/fast/save_ori
#CUDA_VISIBLE_DEVICES=2 python train_full_rl.py --path=$SMYDATA/fast/save_ori/model --abs_dir=$SMYDATA/fast/abstractor_more_baseline/model --ext_dir=$SMYDATA/fast/extractor/model

#rm -r $SMYDATA/fast/save_ori/decoded

#CUDA_VISIBLE_DEVICES=2 python decode_full_model.py --path=$SMYDATA/fast/save_ori/decoded/ --model_dir=$SMYDATA/fast/save_ori/model --beam=5 --test
