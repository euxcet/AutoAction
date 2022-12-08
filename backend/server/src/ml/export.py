import os
import file_utils
from ml.record import Record
from ml.cutter.peak_cutter import PeakCutter
from ml.cutter.random_cutter import RandomCutter
from ml.cutter.positive_cutter import PositiveCutter
from ml.dataset import Dataset
from ml.global_vars import GlobalVars

def create_cutter(cutter_type, model, negative_id):
    if cutter_type['type'] == 'peak':
        return PeakCutter('linear_acc', forward=60, length=GlobalVars.WINDOW_LENGTH,
                          noise=10, skip_index=cutter_type['param'].get('skip_index'))
    elif cutter_type['type'] == 'random':
        return RandomCutter()
    elif cutter_type['type'] == 'positive':
        if model is not None:
            return PositiveCutter(model, negative_id)
    return None

def export_csv(tasklistId, taskIds, trainId, timestamp, cutter_type, dataset_version='0.2', model=None):
    dataset = Dataset()
    tasklist = file_utils.load_taskList_info(tasklistId)
    records = []

    '''
    labels = file_utils.read_labels(file_utils.get_train_label_path(trainId))
    negative_id = -1
    for i in range(len(labels)):
        if labels[i] == 'negative':
            negative_id = i
    '''


    group_id =0
    for group_name in cutter_type:
        for cutter_meta in cutter_type[group_name]:
            cutter = create_cutter(cutter_meta, model, -1)
            if cutter is not None:
                for task in tasklist['tasks']:
                    if task['id'] == cutter_meta['taskId']:
                        taskId = task['id']
                        for subtask in task['subtasks']:
                            subtaskId = subtask['id']
                            recordlist = file_utils.load_recordlist(tasklistId, taskId, subtaskId, dataset_version)
                            for recordId in recordlist:
                                record_path = file_utils.get_record_path(tasklistId, taskId, subtaskId, recordId)
                                if not os.path.exists(record_path): continue
                                for record_filename in os.listdir(record_path):
                                    if record_filename.startswith('Motion') and record_filename.endswith('.bin'):
                                        record_timestamp = record_filename.lstrip('Motion_').rstrip('.bin')
                                        # record filename and timestamp filename
                                        r_fn = os.path.join(record_path, record_filename)
                                        t_fn = os.path.join(record_path, f'Timestamp_{record_timestamp}.json')
                                        for r in range(cutter_meta['repeat']):
                                            records.append(Record(r_fn, t_fn, recordId, group_id, group_name, cutter=cutter))
        group_id += 1
    
    '''
    for task in tasklist['tasks']:
        taskId = task['id']
        if taskId not in taskIds: continue
        # choose all subtasks in selected tasks
        for subtask in task['subtasks']:
            subtaskId = subtask['id']
            # a list of record id
            recordlist = file_utils.load_recordlist(tasklistId, taskId, subtaskId, dataset_version)
            for recordId in recordlist:
                group_name = task['name']
                group_id = taskIds.index(taskId)
                record_path = file_utils.get_record_path(tasklistId, taskId, subtaskId, recordId)
                if not os.path.exists(record_path): continue
                for record_filename in os.listdir(record_path):
                    if record_filename.startswith('Motion') and record_filename.endswith('.bin'):
                        record_timestamp = record_filename.lstrip('Motion_').rstrip('.bin')
                        # record filename and timestamp filename
                        r_fn = os.path.join(record_path, record_filename)
                        t_fn = os.path.join(record_path, f'Timestamp_{record_timestamp}.json')
                        for c in cutter_type[taskId]:
                            cutter = create_cutter(c, model, negative_id)
                            if cutter is not None:
                                for r in range(c['repeat']):
                                    records.append(Record(r_fn, t_fn, recordId, group_id, group_name, cutter=cutter))
    '''

    dataset.insert_records(records) # DONE
    dataset.export_csv(file_utils.get_train_path(trainId))


if __name__ == '__main__':
    pass
