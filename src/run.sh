TRAIN_DATA_PATH=../datasets/synth1/synth1-train.csv
TEST_DATA_PATH=../datasets/synth1/synth1-test.csv
TARGET_COL='Y'
NUM_INDS=50
NUM_GENS=50
MAX_HEIGHT=7
SELECTION_K=2
BETTER_FITNESS='lower' #Or greater
N_CROSS=19
N_MUT=11
P_CROSS=0.9
P_MUT=0.05
SELECTION_TYPE='Roullete' #Or 'Tournament' or 'Lexicase'
NUM_RUNS=30
BASE_NAME='synth1' #or synth2 or concret
python3 main.py --train_data_path  $TRAIN_DATA_PATH --test_data_path $TEST_DATA_PATH --target_col $TARGET_COL \
     --num_inds $NUM_INDS --num_gens $NUM_GENS --max_height $MAX_HEIGHT --selection_k $SELECTION_K \
      --better_fitness $BETTER_FITNESS --elitism --n_cross $N_CROSS --n_mut $N_MUT --p_cross $P_CROSS \
       --p_mut $P_MUT --selection_type $SELECTION_TYPE --num_runs $NUM_RUNS \
       --base_name $BASE_NAME