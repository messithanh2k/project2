import os
import csv
import torch


def export_history(header, value, folder, file_name):
    """ export data to csv format
    Args:
        header (list): headers of the column
        value (list): values of correspoding column
        folder (list): folder path
        file_name: file name with path
    """
    # if folder does not exists make folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_existence = os.path.isfile(file_name)

    # if there is no file make file
    if file_existence == False:
        file = open(file_name, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(value)
    # if there is file overwrite
    else:
        file = open(file_name, 'a', newline='')
        writer = csv.writer(file)
        writer.writerow(value)
    # close file when it is done with writing
    file.close()

def save_checkpoint(checkpoint_path, model, optimizer, epoch):
    state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer, device = 'cpu'):
    print('model loaded from %s' % checkpoint_path)

    state = torch.load(checkpoint_path, map_location=device)
    if model is not None:
        model.load_state_dict(state['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    
    start_epoch = state['epoch']
    return start_epoch