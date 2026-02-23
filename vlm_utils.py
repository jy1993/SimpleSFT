from torch.utils.data import Dataset
import json
import torch
from llm_utils import read_json

class MultiTurnSFTDatasetForVL(Dataset):
	"""docstring for MultiTurnSFTDatasetForVL"""
	def __init__(self, train_filename, processor, only_learn_last=False):
		super(MultiTurnSFTDatasetForVL, self).__init__()
		self.data = read_json(train_filename)
		self.processor = processor
		self.only_learn_last = only_learn_last

	def __getitem__(self, index):
		inputs = self.processor.apply_chat_template(
			self.data[index]['messages'],
			tokenize=True,
			add_generation_prompt=False,
			return_dict=True,
			return_tensors="pt"
		)
		labels = self.get_labels(inputs['input_ids'])
		return inputs['input_ids'], inputs['attention_mask'], inputs['pixel_values'], inputs['image_grid_thw'], labels

	def __len__(self):
		return len(self.data)

	def get_labels(self, input_ids):
		im_start_pos = (input_ids[0] == self.processor.tokenizer.vocab['<|im_start|>']).nonzero().tolist()
		im_end_pos = (input_ids[0] == self.processor.tokenizer.vocab['<|im_end|>']).nonzero().tolist()
		assert len(im_start_pos) == len(im_end_pos)
		labels = [-100] * input_ids.shape[1]
		input_ids_list = input_ids[0].tolist()
		start_offset = 3
		if self.only_learn_last:
			start, end = im_start_pos[-1], im_end_pos[-1]
			labels[start[0]+start_offset:end[0]+2] = input_ids_list[start[0]+start_offset:end[0]+2]
		else:
			for i, (start, end) in enumerate(zip(im_start_pos, im_end_pos)):
				if i > 0 and i % 2 == 0:
					labels[start[0]+start_offset:end[0]+2] = input_ids_list[start[0]+start_offset:end[0]+2]
		return torch.LongTensor(labels).unsqueeze(0)

def pad_and_cat(tensor_list, padding):
	max_len = max([tensor.shape[1] for tensor in tensor_list])
	return torch.cat([torch.cat([tensor, torch.ones(1, max_len - tensor.shape[1], dtype=torch.long) * padding], dim=1) for tensor in tensor_list], dim=0)

def collate_for_sft_vl(batch):
	input_ids = pad_and_cat([item[0] for item in batch], 151643)
	attention_mask = pad_and_cat([item[1] for item in batch], 0)
	pixel_values = torch.cat([item[2] for item in batch], dim=0)
	image_grid_thw = torch.cat([item[3] for item in batch], dim=0)
	labels = pad_and_cat([item[4] for item in batch], -100)
	return input_ids, attention_mask, pixel_values, image_grid_thw, labels

def collate_for_dpo_vl(batch):
	all_input_ids = pad_and_cat([item[0] for item in batch]+[item[4] for item in batch], 151643)
	all_attention_mask = pad_and_cat([item[1] for item in batch]+[item[5] for item in batch], 0)
	all_pixel_values = torch.cat([item[2] for item in batch]+[item[6] for item in batch], dim=0)
	all_image_grid_thw = torch.cat([item[3] for item in batch]+[item[7] for item in batch], dim=0)
	all_labels = pad_and_cat([item[8] for item in batch]+[item[9] for item in batch], -100)
	half = all_input_ids.shape[0] // 2
	return all_input_ids[:half], all_attention_mask[:half], all_pixel_values[:half], all_image_grid_thw[:half], all_labels[:half], all_input_ids[half:], all_attention_mask[half:], all_pixel_values[half:], all_image_grid_thw[half:], all_labels[half:]

def prepare_model_inputs_vl(batch, task_type):
	if task_type == 'sft':
		inputs = {
			'input_ids': batch[0], 
			'attention_mask': batch[1], 
			'pixel_values': batch[2], 
			'image_grid_thw': batch[3], 
			'labels': batch[4]
		}
	elif task_type == 'dpo':
		inputs = {
			'input_ids': torch.cat([batch[0], batch[4]], dim=0),
			'attention_mask': torch.cat([batch[1], batch[5]], dim=0),
			'pixel_values': torch.cat([batch[2], batch[6]], dim=0),
			'image_grid_thw': torch.cat([batch[3], batch[7]], dim=0),
			'chosen_labels': batch[8],
			'rejected_labels': batch[9]
		}
	return inputs