# IMCPT-SparseGM-generator
This generator creates IMCPT-SparseGM based on Image_Matching_Challange_Data.

Note that you should install colmap and download Image_Matching_Challange_Data before you create IMCPT-SparseGM by just running 
    
    python dataset_generator.py

Arguments are the following (All have defaults):

    --root             'source dataset directory'
    --out_dir          'output dataset directory'
    --pt_num           'universal point number to be selected'
    --min_exist_num    'min num of img an anchor exists in'
    --dis_rate         'min distance rate when selecting points'
    --exist_dis_rate   'min distance rate when judging anchors\' existence'
