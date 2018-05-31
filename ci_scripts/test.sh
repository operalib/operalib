set -e

# Get into a temp directory to run test from the installed operalib and
# check if we do not leave artifacts
mkdir -p $TEST_DIR

cd $TEST_DIR

if [[ "$RUN_FLAKE8" == "true" ]]; then
    flake8 -v
fi
if [[ "$COVERAGE" == "true" ]]; then
    py.test -v -x --ignore=setup.py --pyargs $MODULE --cov=$MODULE
else
    py.test -v -x --ignore=setup.py --pyargs $MODULE
fi
