import os
import file_utils
from ml.record import Record
from ml.cutter.peak_cutter import PeakCutter
from ml.cutter.random_cutter import RandomCutter
from ml.dataset import Dataset

def export_csv(tasklistId, taskIds, trainId, timestamp):
    dataset = Dataset()
    cutter_dict = {
        'peak': [PeakCutter(0)],
        'random': [RandomCutter(), RandomCutter()],
        'peakrandom': [RandomCutter(), PeakCutter(0)]
    }
    tasklist = file_utils.load_taskList_info(tasklistId)
    records = []
    for task in tasklist['task']:
        taskId = task['id']
        if taskId in taskIds:
            for subtask in task['subtask']:
                subtaskId = subtask['id']
                recordlist = file_utils.load_recordlist(tasklistId, taskId, subtaskId)
                for recordId in recordlist:
                    group_name = task['name']
                    group_id = taskIds.index(taskId)
                    cutter_type = 'peak' # TODO

                    record_path = file_utils.get_record_path(tasklistId, taskId, subtaskId, recordId)

                    if os.path.exists(record_path):
                        for record_filename in os.listdir(record_path):
                            if record_filename.startswith('Sensor') and record_filename.endswith('json'):
                                r_fn = os.path.join(record_path, record_filename)
                                t_fn = os.path.join(record_path, 'Timestamp' + record_filename[6:])
                                for c in cutter_dict[cutter_type]:
                                    records.append(Record(r_fn, t_fn, recordId, group_id, group_name, cutter=c))


    dataset.insert_records(records)
    dataset.export_csv(file_utils.get_train_path(trainId))
    '''
    for record_folder in os.listdir(dataset_root):
        if not record_folder.startswith('.'): # ignore .DS_Store
            if len(record_folder.split('_')) == 4: # format: {group_name}:{group_id}:{cutter}:{description}
                group_name = record_folder.split('_')[0]
                group_id = int(record_folder.split('_')[1])
                cutter_type = record_folder.split('_')[2]
                description = record_folder.split('_')[3]

                folder_name = os.path.join(dataset_root, record_folder)
                for record_filename in os.listdir(folder_name):
                    if record_filename.startswith('Sensor') and record_filename.endswith('json'):
                        r_fn = os.path.join(folder_name, record_filename)
                        t_fn = os.path.join(folder_name, 'Timestamp' + record_filename[6:])
                        for c in cutter_dict[cutter_type]:
                            records.append(Record(r_fn, t_fn, group_id, group_name, description, c))
    '''


'''

if __name__ == '__main__':
    records = load_record()
    dataset = Dataset()
    dataset.insert_records(records)
    dataset.export_csv('../output/')

'''