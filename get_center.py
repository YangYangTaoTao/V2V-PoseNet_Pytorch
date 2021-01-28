from estimate_center.util import ITOPCenterDataset
from estimate_center.util import estimate_points
from estimate_center.model import resnet18
import torch
import os
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from lib.solver import train_epoch, val_epoch, test_epoch
# test

data_dir = './datasets/depthmap'
center_dir = './datasets/center/ITOP_center'
db = 'side'
batch_size = 128
num_workers = 16
save_checkpoint = True
save_model_path = "estimate_center/pretrain"
start_epoch = 0
epochs_num = 100
lr_patience = 10
checkpoint_num_epochs = 10
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float

train_set = ITOPCenterDataset(root=data_dir, center_dir=center_dir, db=db, mode='train')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_set = ITOPCenterDataset(root=data_dir, center_dir=center_dir, db=db, mode='test')
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

model = resnet18()
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.1)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=lr_patience)

# Train and Validate
# print('==> Training ..')
# for epoch in range(start_epoch, start_epoch + epochs_num):
#     print('Epoch: {}'.format(epoch))
#     train_epoch(model, criterion, optimizer, train_loader, device=device, dtype=dtype)
#     val_loss = val_epoch(model, criterion, val_loader, device=device, dtype=dtype)
#     print(val_loss)
#     scheduler.step(val_loss)
#     if save_checkpoint and epoch % checkpoint_num_epochs == 0:
#         if not os.path.exists(save_model_path):
#             os.mkdir(save_model_path)
#         checkpoint_file = os.path.join(save_model_path, 'epoch'+str(epoch)+'.pth')
#         checkpoint = {
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'epoch': epoch
#         }
#         torch.save(checkpoint, checkpoint_file)

class BatchResultCollector():
    def __init__(self, samples_num, transform_output):
        self.samples_num = samples_num
        self.transform_output = transform_output
        self.keypoints = None
        self.idx = 0

    def __call__(self, data_batch):
        inputs_batch, outputs_batch, extra_batch = data_batch
        outputs_batch = outputs_batch.cpu().numpy()
        refpoints_batch = extra_batch.cpu().numpy()

        keypoints_batch = self.transform_output(outputs_batch, refpoints_batch)

        if self.keypoints is None:
            # Initialize keypoints until dimensions awailable now
            self.keypoints = np.zeros((self.samples_num, *keypoints_batch.shape[1:]))

        batch_size = keypoints_batch.shape[0] 
        self.keypoints[self.idx:self.idx+batch_size] = keypoints_batch
        self.idx += batch_size

    def get_result(self):
        return self.keypoints


def save_keypoints(filename, keypoints):
    # Reshape one sample keypoints into one line
    keypoints = keypoints.reshape(keypoints.shape[0], -1)
    np.savetxt(filename, keypoints, fmt='%0.4f')
# get train/test center point
savepath = "estimate_center/my_train_center.txt"
pretrain = torch.load("estimate_center/pretrain/epoch90.pth")
ref_pt_file = os.path.join("ITOP_" + str(db) + "_view", "center_train.txt")
model.load_state_dict(pretrain['model_state_dict'])
model = model.eval()
# val_loss = val_epoch(model, criterion, val_loader, device=device, dtype=dtype)
# print(val_loss)
estimate_points(data_dir, db, 'train', model, device, center_dir, ref_pt_file, savepath)


