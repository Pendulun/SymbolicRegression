TRAIN_DATA_PATH=../datasets/synth1/synth1-train.csv
TEST_DATA_PATH=../datasets/synth1/synth1-test.csv
TARGET_COL='Y'
NUM_INDS=100
NUM_GENS=100
MAX_HEIGHT=7
SELECTION_K=2
BETTER_FITNESS='lower' #Or greater
N_CROSS=40
N_MUT=19
P_CROSS=0.6
P_MUT=0.3
SELECTION_TYPE='Roullete' #Or 'Tournament' or 'Lexicase'
RANDOM_SEED=42
NUM_RUNS=10
BASE_NAME='synth1' #or synth2 or concret
python3 main.py --train_data_path  $TRAIN_DATA_PATH --test_data_path $TEST_DATA_PATH --target_col $TARGET_COL \
     --num_inds $NUM_INDS --num_gens $NUM_GENS --max_height $MAX_HEIGHT --selection_k $SELECTION_K \
      --better_fitness $BETTER_FITNESS --elitism --n_cross $N_CROSS --n_mut $N_MUT --p_cross $P_CROSS \
       --p_mut $P_MUT --selection_type $SELECTION_TYPE --random_seed $RANDOM_SEED --num_runs $NUM_RUNS \
       --base_name $BASE_NAME