python train.py \
    --id='Markov' \
    --train=./data/random-walk-train.pkl \
    --dev=./data/random-walk-dev.pkl \
    -m

python train.py \
    --id='X' \
    --train=./data/random-walk-train.pkl \
    --dev=./data/random-walk-dev.pkl \
    -x=count

python train.py \
    --id='Z=2' \
    --train=./data/random-walk-train.pkl \
    --dev=./data/random-walk-dev.pkl \
    -z=2

python train.py \
    --id='Z=10' \
    --train=./data/random-walk-train.pkl \
    --dev=./data/random-walk-dev.pkl \
    -z=10

python train.py \
    --id='Markov_X' \
    --train=./data/random-walk-train.pkl \
    --dev=./data/random-walk-dev.pkl \
    -m \
    -x=count

python train.py \
    --id='Markov_Z=2' \
    --train=./data/random-walk-train.pkl \
    --dev=./data/random-walk-dev.pkl \
    -m \
    -z=2

python train.py \
    --id='Markov_Z=10' \
    --train=./data/random-walk-train.pkl \
    --dev=./data/random-walk-dev.pkl \
    -m \
    -z=10

python train.py \
    --id='Markov_X_Z=2' \
    --train=./data/random-walk-train.pkl \
    --dev=./data/random-walk-dev.pkl \
    -m \
    -x=count \
    -z=2

