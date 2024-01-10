#!/bin/bash
#!/usr/bin/env python

eval "$(conda shell.bash hook)"
conda activate seal_ad

nohup bash run_device.sh '' mlc bibtex 'base' 32 &> mlc2.out &
BACK_PID=$!
wait $BACK_PID
nohup bash run_device.sh '' mlc deli 'base' 32 &> mlc3.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'base' 16 &> mlc2.out &
BACK_PID=$!
wait $BACK_PID
nohup bash run_device.sh '' mlc deli 'base' 16 &> mlc2.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc bibtex 'nodistil' '' 'label_smoothing' '0.9' '0.1' 32 64 &> mlc4.out &
BACK_PID=$!
wait $BACK_PID
nohup bash run_device.sh '' mlc bibtex 'hard_label' '' 'label_smoothing' '0.9' '0.1' 32 64 &> mlc5.out &
BACK_PID=$!
wait $BACK_PID
nohup bash run_device.sh '' mlc bibtex 'soft_label' '' 'label_smoothing' '0.9' '0.1' 32 64 &> mlc6.out &
BACK_PID=$!
wait $BACK_PID
nohup bash run_device.sh '' mlc bibtex 'label_smoothing' '0.1' 'nodistil' '' '0.1' 32 64 &> mlc7.out &
BACK_PID=$!
wait $BACK_PID
nohup bash run_device.sh '' mlc bibtex 'label_smoothing' '0.1' 'label_smoothing' '0.9' '0.1' 32 64 &> mlc8.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc bibtex 'hard_label' '' 'label_smoothing' '0.9' '0.1' 32 128 &> mlc9.out &
BACK_PID=$!
wait $BACK_PID
nohup bash run_device.sh '' mlc bibtex 'label_smoothing' '0.1' 'nodistil' '' '0.1' 32 128 &> mlc10.out &
BACK_PID=$!
wait $BACK_PID
nohup bash run_device.sh '' mlc bibtex 'label_smoothing' '0.1' 'label_smoothing' '0.9' '0.1' 32 128 &> mlc11.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.1' 'nodistil' '' '0.1' 32 64 &> mlc12.out &
BACK_PID=$!
wait $BACK_PID
nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.1' 'label_smoothing' '0.9' '0.1' 32 128 &> mlc13.out &
BACK_PID=$!
wait $BACK_PID

############################################
nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'nodistil' '' '0.1' 32 32 &> mlc1.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'hard_label' '' '0.1' 32 32 &> mlc2.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'label_smoothing' '0.1' '0.1' 32 32 &> mlc3.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'label_smoothing' '0.9' '0.1' 32 32 &> mlc4.out &
BACK_PID=$!
wait $BACK_PID

###############################
nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'nodistil' '' '0.1' 32 32 &> mlc5.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'hard_label' '' '0.1' 32 32 &> mlc6.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'label_smoothing' '0.1' '0.1' 32 32 &> mlc7.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'label_smoothing' '0.9' '0.1' 32 32 &> mlc8.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'nodistil' '' '0.1' 32 32 &> mlc9.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'hard_label' '' '0.1' 32 32 &> mlc10.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'label_smoothing' '0.1' '0.1' 32 32 &> mlc11.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'label_smoothing' '0.9' '0.1' 32 32 &> mlc12.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'nodistil' '' '0.1' 32 32 &> mlc13.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'hard_label' '' '0.1' 32 32 &> mlc14.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'label_smoothing' '0.1' '0.1' 32 32 &> mlc15.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'label_smoothing' '0.9' '0.1' 32 32 &> mlc16.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'nodistil' '' '0.1' 32 32 &> mlc17.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'hard_label' '' '0.1' 32 32 &> mlc18.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'label_smoothing' '0.1' '0.1' 32 32 &> mlc19.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'label_smoothing' '0.9' '0.1' 32 32 &> mlc20.out &
BACK_PID=$!
wait $BACK_PID

###############################
###############################
nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'nodistil' '' '0.1' 32 64 &> mlc1.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'hard_label' '' '0.1' 32 64 &> mlc2.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'label_smoothing' '0.1' '0.1' 32 64 &> mlc3.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'label_smoothing' '0.9' '0.1' 32 64 &> mlc4.out &
BACK_PID=$!
wait $BACK_PID

###############################
nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'nodistil' '' '0.1' 32 64 &> mlc5.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'hard_label' '' '0.1' 32 64 &> mlc6.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'label_smoothing' '0.1' '0.1' 32 64 &> mlc7.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'label_smoothing' '0.9' '0.1' 32 64 &> mlc8.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'nodistil' '' '0.1' 32 64 &> mlc9.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'hard_label' '' '0.1' 32 64 &> mlc10.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'label_smoothing' '0.1' '0.1' 32 64 &> mlc11.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'label_smoothing' '0.9' '0.1' 32 64 &> mlc12.out &
BACK_PID=$!
wait $BACK_PID

################################
nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'nodistil' '' '0.1' 32 64 &> mlc13.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'hard_label' '' '0.1' 32 64 &> mlc14.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'label_smoothing' '0.1' '0.1' 32 64 &> mlc15.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'label_smoothing' '0.9' '0.1' 32 64 &> mlc16.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'nodistil' '' '0.1' 32 64 &> mlc17.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'hard_label' '' '0.1' 32 64 &> mlc18.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'label_smoothing' '0.1' '0.1' 32 64 &> mlc19.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'label_smoothing' '0.9' '0.1' 32 64 &> mlc20.out &
BACK_PID=$!
wait $BACK_PID

###############################
###############################
nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'nodistil' '' '0.1' 32 128 &> mlc1.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'hard_label' '' '0.1' 32 128 &> mlc2.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'label_smoothing' '0.1' '0.1' 32 128 &> mlc3.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'label_smoothing' '0.9' '0.1' 32 128 &> mlc4.out &
BACK_PID=$!
wait $BACK_PID

###############################
nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'nodistil' '' '0.1' 32 128 &> mlc5.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'hard_label' '' '0.1' 32 128 &> mlc6.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'label_smoothing' '0.1' '0.1' 32 128 &> mlc7.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'label_smoothing' '0.9' '0.1' 32 128 &> mlc8.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'nodistil' '' '0.1' 32 128 &> mlc9.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'hard_label' '' '0.1' 32 128 &> mlc10.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'label_smoothing' '0.1' '0.1' 32 128 &> mlc11.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'label_smoothing' '0.9' '0.1' 32 128 &> mlc12.out &
BACK_PID=$!
wait $BACK_PID

################################
nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'nodistil' '' '0.1' 32 128 &> mlc13.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'hard_label' '' '0.1' 32 128 &> mlc14.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'label_smoothing' '0.1' '0.1' 32 128 &> mlc15.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'label_smoothing' '0.9' '0.1' 32 128 &> mlc16.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'nodistil' '' '0.1' 32 128 &> mlc17.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'hard_label' '' '0.1' 32 128 &> mlc18.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'label_smoothing' '0.1' '0.1' 32 128 &> mlc19.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'label_smoothing' '0.9' '0.1' 32 128 &> mlc20.out &
BACK_PID=$!
wait $BACK_PID

###########################################
####    END OF CAL500 batch size 32   #####
###########################################

############################################
nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'nodistil' '' '0.1' 16 16 &> mlc1.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'hard_label' '' '0.1' 16 16 &> mlc2.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'label_smoothing' '0.1' '0.1' 16 16 &> mlc3.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'label_smoothing' '0.9' '0.1' 16 16 &> mlc4.out &
BACK_PID=$!
wait $BACK_PID

###############################
nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'nodistil' '' '0.1' 16 16 &> mlc5.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'hard_label' '' '0.1' 16 16 &> mlc6.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'label_smoothing' '0.1' '0.1' 16 16 &> mlc7.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'label_smoothing' '0.9' '0.1' 16 16 &> mlc8.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'nodistil' '' '0.1' 16 16 &> mlc9.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'hard_label' '' '0.1' 16 16 &> mlc10.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'label_smoothing' '0.1' '0.1' 16 16 &> mlc11.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'label_smoothing' '0.9' '0.1' 16 16 &> mlc12.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'nodistil' '' '0.1' 16 16 &> mlc13.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'hard_label' '' '0.1' 16 16 &> mlc14.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'label_smoothing' '0.1' '0.1' 16 16 &> mlc15.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'label_smoothing' '0.9' '0.1' 16 16 &> mlc16.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'nodistil' '' '0.1' 16 16 &> mlc17.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'hard_label' '' '0.1' 16 16 &> mlc18.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'label_smoothing' '0.1' '0.1' 16 16 &> mlc19.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'label_smoothing' '0.9' '0.1' 16 16 &> mlc20.out &
BACK_PID=$!
wait $BACK_PID

###############################
###############################
nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'nodistil' '' '0.1' 16 32 &> mlc1.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'hard_label' '' '0.1' 16 32 &> mlc2.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'label_smoothing' '0.1' '0.1' 16 32 &> mlc3.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'label_smoothing' '0.9' '0.1' 16 32 &> mlc4.out &
BACK_PID=$!
wait $BACK_PID

###############################
nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'nodistil' '' '0.1' 16 32 &> mlc5.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'hard_label' '' '0.1' 16 32 &> mlc6.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'label_smoothing' '0.1' '0.1' 16 32 &> mlc7.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'label_smoothing' '0.9' '0.1' 16 32 &> mlc8.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'nodistil' '' '0.1' 16 32 &> mlc9.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'hard_label' '' '0.1' 16 32 &> mlc10.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'label_smoothing' '0.1' '0.1' 16 32 &> mlc11.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'label_smoothing' '0.9' '0.1' 16 32 &> mlc12.out &
BACK_PID=$!
wait $BACK_PID

################################
nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'nodistil' '' '0.1' 16 32 &> mlc13.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'hard_label' '' '0.1' 16 32 &> mlc14.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'label_smoothing' '0.1' '0.1' 16 32 &> mlc15.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'label_smoothing' '0.9' '0.1' 16 32 &> mlc16.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'nodistil' '' '0.1' 16 32 &> mlc17.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'hard_label' '' '0.1' 16 32 &> mlc18.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'label_smoothing' '0.1' '0.1' 16 32 &> mlc19.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'label_smoothing' '0.9' '0.1' 16 32 &> mlc20.out &
BACK_PID=$!
wait $BACK_PID

###############################
###############################
nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'nodistil' '' '0.1' 16 64 &> mlc1.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'hard_label' '' '0.1' 16 64 &> mlc2.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'label_smoothing' '0.1' '0.1' 16 64 &> mlc3.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'nodistil' '' 'label_smoothing' '0.9' '0.1' 16 64 &> mlc4.out &
BACK_PID=$!
wait $BACK_PID

###############################
nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'nodistil' '' '0.1' 16 64 &> mlc5.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'hard_label' '' '0.1' 16 64 &> mlc6.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'label_smoothing' '0.1' '0.1' 16 64 &> mlc7.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'hard_label' '' 'label_smoothing' '0.9' '0.1' 16 64 &> mlc8.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'nodistil' '' '0.1' 16 64 &> mlc9.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'hard_label' '' '0.1' 16 64 &> mlc10.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'label_smoothing' '0.1' '0.1' 16 64 &> mlc11.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'soft_label' '' 'label_smoothing' '0.9' '0.1' 16 64 &> mlc12.out &
BACK_PID=$!
wait $BACK_PID

################################
nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'nodistil' '' '0.1' 16 64 &> mlc13.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'hard_label' '' '0.1' 16 64 &> mlc14.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'label_smoothing' '0.1' '0.1' 16 64 &> mlc15.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.1' 'label_smoothing' '0.9' '0.1' 16 64 &> mlc16.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'nodistil' '' '0.1' 16 64 &> mlc17.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'hard_label' '' '0.1' 16 64 &> mlc18.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'label_smoothing' '0.1' '0.1' 16 64 &> mlc19.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc cal500 'label_smoothing' '0.9' 'label_smoothing' '0.9' '0.1' 16 64 &> mlc20.out &
BACK_PID=$!
wait $BACK_PID

###########################################
####    END OF CAL500 batch size 16   #####
###########################################
############################################
nohup bash run_device.sh '' mlc deli 'nodistil' '' 'nodistil' '' '0.1' 16 16 &> mlc1.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'nodistil' '' 'hard_label' '' '0.1' 16 16 &> mlc2.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'nodistil' '' 'label_smoothing' '0.1' '0.1' 16 16 &> mlc3.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'nodistil' '' 'label_smoothing' '0.9' '0.1' 16 16 &> mlc4.out &
BACK_PID=$!
wait $BACK_PID

###############################
nohup bash run_device.sh '' mlc deli 'hard_label' '' 'nodistil' '' '0.1' 16 16 &> mlc5.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'hard_label' '' 'hard_label' '' '0.1' 16 16 &> mlc6.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'hard_label' '' 'label_smoothing' '0.1' '0.1' 16 16 &> mlc7.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'hard_label' '' 'label_smoothing' '0.9' '0.1' 16 16 &> mlc8.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc deli 'soft_label' '' 'nodistil' '' '0.1' 16 16 &> mlc9.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'soft_label' '' 'hard_label' '' '0.1' 16 16 &> mlc10.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'soft_label' '' 'label_smoothing' '0.1' '0.1' 16 16 &> mlc11.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'soft_label' '' 'label_smoothing' '0.9' '0.1' 16 16 &> mlc12.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.1' 'nodistil' '' '0.1' 16 16 &> mlc13.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.1' 'hard_label' '' '0.1' 16 16 &> mlc14.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.1' 'label_smoothing' '0.1' '0.1' 16 16 &> mlc15.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.1' 'label_smoothing' '0.9' '0.1' 16 16 &> mlc16.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.9' 'nodistil' '' '0.1' 16 16 &> mlc17.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.9' 'hard_label' '' '0.1' 16 16 &> mlc18.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.9' 'label_smoothing' '0.1' '0.1' 16 16 &> mlc19.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.9' 'label_smoothing' '0.9' '0.1' 16 16 &> mlc20.out &
BACK_PID=$!
wait $BACK_PID

###############################
###############################
nohup bash run_device.sh '' mlc deli 'nodistil' '' 'nodistil' '' '0.1' 16 32 &> mlc1.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'nodistil' '' 'hard_label' '' '0.1' 16 32 &> mlc2.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'nodistil' '' 'label_smoothing' '0.1' '0.1' 16 32 &> mlc3.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'nodistil' '' 'label_smoothing' '0.9' '0.1' 16 32 &> mlc4.out &
BACK_PID=$!
wait $BACK_PID

###############################
nohup bash run_device.sh '' mlc deli 'hard_label' '' 'nodistil' '' '0.1' 16 32 &> mlc5.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'hard_label' '' 'hard_label' '' '0.1' 16 32 &> mlc6.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'hard_label' '' 'label_smoothing' '0.1' '0.1' 16 32 &> mlc7.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'hard_label' '' 'label_smoothing' '0.9' '0.1' 16 32 &> mlc8.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc deli 'soft_label' '' 'nodistil' '' '0.1' 16 32 &> mlc9.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'soft_label' '' 'hard_label' '' '0.1' 16 32 &> mlc10.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'soft_label' '' 'label_smoothing' '0.1' '0.1' 16 32 &> mlc11.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'soft_label' '' 'label_smoothing' '0.9' '0.1' 16 32 &> mlc12.out &
BACK_PID=$!
wait $BACK_PID

################################
nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.1' 'nodistil' '' '0.1' 16 32 &> mlc13.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.1' 'hard_label' '' '0.1' 16 32 &> mlc14.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.1' 'label_smoothing' '0.1' '0.1' 16 32 &> mlc15.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.1' 'label_smoothing' '0.9' '0.1' 16 32 &> mlc16.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.9' 'nodistil' '' '0.1' 16 32 &> mlc17.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.9' 'hard_label' '' '0.1' 16 32 &> mlc18.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.9' 'label_smoothing' '0.1' '0.1' 16 32 &> mlc19.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.9' 'label_smoothing' '0.9' '0.1' 16 32 &> mlc20.out &
BACK_PID=$!
wait $BACK_PID

###############################
###############################
nohup bash run_device.sh '' mlc deli 'nodistil' '' 'nodistil' '' '0.1' 16 64 &> mlc1.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'nodistil' '' 'hard_label' '' '0.1' 16 64 &> mlc2.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'nodistil' '' 'label_smoothing' '0.1' '0.1' 16 64 &> mlc3.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'nodistil' '' 'label_smoothing' '0.9' '0.1' 16 64 &> mlc4.out &
BACK_PID=$!
wait $BACK_PID

###############################
nohup bash run_device.sh '' mlc deli 'hard_label' '' 'nodistil' '' '0.1' 16 64 &> mlc5.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'hard_label' '' 'hard_label' '' '0.1' 16 64 &> mlc6.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'hard_label' '' 'label_smoothing' '0.1' '0.1' 16 64 &> mlc7.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'hard_label' '' 'label_smoothing' '0.9' '0.1' 16 64 &> mlc8.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc deli 'soft_label' '' 'nodistil' '' '0.1' 16 64 &> mlc9.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'soft_label' '' 'hard_label' '' '0.1' 16 64 &> mlc10.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'soft_label' '' 'label_smoothing' '0.1' '0.1' 16 64 &> mlc11.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'soft_label' '' 'label_smoothing' '0.9' '0.1' 16 64 &> mlc12.out &
BACK_PID=$!
wait $BACK_PID

################################
nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.1' 'nodistil' '' '0.1' 16 64 &> mlc13.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.1' 'hard_label' '' '0.1' 16 64 &> mlc14.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.1' 'label_smoothing' '0.1' '0.1' 16 64 &> mlc15.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.1' 'label_smoothing' '0.9' '0.1' 16 64 &> mlc16.out &
BACK_PID=$!
wait $BACK_PID

################################3
nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.9' 'nodistil' '' '0.1' 16 64 &> mlc17.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.9' 'hard_label' '' '0.1' 16 64 &> mlc18.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.9' 'label_smoothing' '0.1' '0.1' 16 64 &> mlc19.out &
BACK_PID=$!
wait $BACK_PID

nohup bash run_device.sh '' mlc deli 'label_smoothing' '0.9' 'label_smoothing' '0.9' '0.1' 16 64 &> mlc20.out &
BACK_PID=$!
wait $BACK_PID

###########################################
####    END OF deli batch size 16   #####
###########################################


exit
