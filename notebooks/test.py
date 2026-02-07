# %%
from trajdata import AgentBatch, UnifiedDataset

# %%
# data_dir = "../data/waymo_motion_v1_3_0/scenario/"
# file_list = tf.io.gfile.glob(os.path.join(data_dir, '*'))
# print(f"Found {len(file_list)} files.")

# %%
dataset = UnifiedDataset(
    desired_data=["waymo_train"],
    data_dirs={  # Remember to change this to match your filesystem!
        "waymo_train": "../data/waymo_motion_v1_3_0/scenario/"
    },
)

# %%



