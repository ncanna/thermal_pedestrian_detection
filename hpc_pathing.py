import os, time

# Check what the main folder is on rivanna
main_folder = os.path.dirname(os.path.abspath(__file__))
time.sleep(5)
print(main_folder)
# os.makedirs("/scratch/na3au/modelRuns")