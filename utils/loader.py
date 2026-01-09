import pandas as pd
import os 
import json
def load_dataset(dataset, file_path, video_root):
    items = []
    if "omninext" in dataset:
        with open(data_file, "r") as fr:
            data = json.load(fr, strict=False)
        for sample in data:
            item = {}
            item['id'] = sample['id']
            item['video'] = f"{video_root}/{sample['original_video']}"
            item['question'] = sample['question']
            item['answer'] = sample['answer']
            item['options'] = sample['options']
                
            items.append(item)
    elif dataset == "worldsense":
        df = pd.read_csv(data_file, sep="\t")
        for row in df.iterrows():
            item = {}
            idx, row = row
            row = row.to_dict()
            video_path = row['video_path']
            domain = row['domain']
            options = eval(row['candidates'])
            question = row['question']
            answer = row['answer']
            item['video'] = video_path.replace("./",video_root)
            item['domain'] = domain
            item['options'] = options
            item['question'] = question
            item['answer'] = answer
            items.append(item)
    elif dataset == "dailyomni":
        with open(data_file, "r") as fr:
            df = json.load(fr)
        for sample in df:
            item = {}
            item['video'] = video_root + "/" + 'videos/' + sample['video_id'] + '/' + sample['video_id'] + '_video' + ".mp4"
            item['domain'] = None
            item['question'] = sample['Question']
            item['answer'] = sample['Answer']
            item['options'] = sample['Choice']
            items.append(item)
    elif dataset == 'omnivideobench':
        with open(data_file, "r") as fr:
            data = json.load(fr)
        for sample in data:
            video_path = video_root + '/' + sample['video'] + ".mp4"
            for sample_question in sample['questions']:
                
                item = {}
                item['video'] = video_path
                item['domain'] = sample_question['question_type']
                item['answer'] = sample_question['correct_option']
                item['options'] = sample_question['options']
                item['question'] = sample_question['question']
                items.append(item)
    elif dataset == 'jointavbench':
        with open(data_file, "r") as fr:
            data = json.load(fr)
        for sample in data:
            item = {}
            
            video_path = video_root + "/" + "jointavbench" + "/" + sample['qid'] + ".mp4"
    
            item['domain'] = sample['task']
            item['question'] = sample['question']
            item['options'] = []
            gt = sample['correct_answer']
            gt_alpha = None
            assert len(sample['options']) == 4
            for alpha, option in zip(['A.','B.','C.','D.'], sample['options']):
                item['options'].append(alpha + option)
                if gt == option:
                    gt_alpha = alpha.strip('.')
            assert gt_alpha is not None
            item['answer'] = gt_alpha
            items.append(item)
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented!")
    return items