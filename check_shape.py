import numpy as np

def check():
    path = r"C:\Users\ayoub\Desktop\Stage\Project_bruit_modifie\data\GIGADATASET_COLAB_NPZ\train_cohort3_polamb_nonzero.npz"
    d = np.load(path, mmap_mode='r')
    print("X shape:", d["X"].shape)
    print("M shape:", d["M"].shape)
    
if __name__ == "__main__":
    check()
