import os

files_to_remove = [

    # =====================================
    # Generated CSV files
    # =====================================
    "results_mnist_cc.csv"
    #"pareto_front_mnist.csv",
    #"aer_summary_dncnn.csv",
    

    # =====================================
    # Generated plots
    # =====================================
    
    
]

print("\nRemoving generated outputs...\n")

for file in files_to_remove:

    if os.path.exists(file):

        os.remove(file)

        print(f"Deleted: {file}")

    else:
        print(f"Not found: {file}")

print("\nCleanup complete.")