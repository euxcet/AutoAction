import os
from record import Record
from cutter.peak_cutter import PeakCutter
from cutter.random_cutter import RandomCutter
from dataset import Dataset

def load_record():
    cutter_dict = {
        'peak': [PeakCutter(0)],
        'random': [RandomCutter(), RandomCutter()],
        'peakrandom': [RandomCutter(), PeakCutter(0)]
    }
    dataset_root = '../data'
    records = []
    for record_folder in os.listdir(dataset_root):
        if not record_folder.startswith('.'): # ignore .DS_Store
            if len(record_folder.split('_')) == 4: # format: {group_name}:{group_id}:{cutter}:{description}
                #if not record_folder.startswith('draw'):
                #    continue
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
    return records

if __name__ == '__main__':
    records = load_record()
    dataset = Dataset()
    dataset.insert_records(records)
    dataset.export_csv('../output/')
