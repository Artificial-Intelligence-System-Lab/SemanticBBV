root=$1

# find all *.bb files and only collect their name into a list
declare -a benchmarks

# Read all .bb files (only filenames, without path and extension) into array LIST
mapfile -t benchmarks < <(
  find "$root" -type f -name '*.bb' \
    -exec basename {} .bb \;
)

# print the list
echo $benchmarks

for benchmark in ${benchmarks[@]}; do
    # find the corresponding *.bbv file
    bbv_file=$(find $root -name "$benchmark.bb")
    output_file=$root/$benchmark.bb.m
    map_file=$root/$benchmark.vectors.clusters.pkl
    # run the python script
    python3 simpoint_bbv_modifier.py -i $bbv_file -m $map_file -o $output_file
done