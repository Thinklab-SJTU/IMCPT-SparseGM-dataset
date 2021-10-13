# IMCPT-SparseGM-generator
This generator creates IMCPT-SparseGM based on Image_Matching_Challange_Data.

Note that you should install colmap and download Image_Matching_Challange_Data before you create IMCPT-SparseGM by just running 
    
    python dataset_generator.py

Arguments are the following:

    --root             'source dataset directory'                             default='/mnt/nas/dataset_share/Image_Matching_Challange_Data'
    --out_dir          'output dataset directory'                             default='picture'
    --pt_num           'universal point number to be selected'                default=50
    --min_exist_num    'min num of img an anchor exists in'                   default=10
    --dis_rate         'min distance rate when selecting points'              default=1.0
    --exist_dis_rate   'min distance rate when judging anchors\' existence'   default=0.75
