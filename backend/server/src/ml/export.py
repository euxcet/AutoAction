import os
import file_utils
from ml.record import Record
from ml.cutter.peak_cutter import PeakCutter
from ml.cutter.random_cutter import RandomCutter
from ml.dataset import Dataset
from ml.global_vars import GlobalVars

def export_csv(tasklistId, taskIds, trainId, timestamp, dataset_version='0.2'):
    dataset = Dataset()
    cutter_dict = {
        'peak': [PeakCutter('linear_acc', forward=100,
            length=GlobalVars.WINDOW_LENGTH, noise=20)],
        'random': [RandomCutter(), RandomCutter()],
        'peakrandom': [RandomCutter(), PeakCutter('linear_acc', forward=100,
            length=GlobalVars.WINDOW_LENGTH, noise=20)]
    }
    tasklist = file_utils.load_taskList_info(tasklistId)
    records = []
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
                cutter_type = 'peak'
                record_path = file_utils.get_record_path(tasklistId, taskId, subtaskId, recordId)
                if not os.path.exists(record_path): continue
                for record_filename in os.listdir(record_path):
                    if record_filename.startswith('Motion') and record_filename.endswith('.bin'):
                        record_timestamp = record_filename.lstrip('Motion_').rstrip('.bin')
                        # record filename and timestamp filename
                        r_fn = os.path.join(record_path, record_filename)
                        t_fn = os.path.join(record_path, f'Timestamp_{record_timestamp}.json')
                        for c in cutter_dict[cutter_type]:
                            records.append(Record(r_fn, t_fn, recordId, group_id, group_name, cutter=c))

    dataset.insert_records(records) # DONE
    dataset.export_csv(file_utils.get_train_path(trainId))


if __name__ == '__main__':
    pass
