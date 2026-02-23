import pandas as pd
import base64
import json
import os
from llm_utils import to_json
import numpy as np

def get_system(ds_name):
	if ds_name == 'android_control':
		system = (
			'You are an expert in controlling mobile devices.'
			'To complete the task, at each step, your need to choose the right tool.\n\n'
			'# Output format:\n'
			'<think>...</think>\n'
			'<tool_call>...</tool_call>\n\n'
			'# Available tools:\n'
			'- click(x, y): click the point on the screen with specified (x, y) coordinates\n'
			'- long_press(x, y): press the point on the screen with specified (x, y) coordinates for a few seconds\n'
			'- scroll(direction): scroll to the specified direction: "up", "down", "left", or "right"\n'
			'- input_text(text): input the specified text into the input inbox\n'
			'- navigate_home(): go to home page\n'
			'- navigate_back(): go to previous page\n'
			'- open_app(app_name): launch the app\n'
			'- wait(): wait for a few second for the changes to occur\n'
			'- finish(): finish the task\n'
			'\n'
			'# Example output:\n'
			'<think>...</think>\n'
			'<tool_call>{"name": "scroll", "arguments": {"direction": up"}}</tool_call>\n\n'
			'Note:\n'
			'- Plan the task and explain your reasoning in "think" part\n'
			'- Choose the right tool from the given tools\n\n'
			'# Task:\n'
		)
	return system

def process_history(messages, num_images):
	new_messages = []
	image_cnt = 0
	split_index = len(messages) - 2 * num_images
	for idx, item in enumerate(messages):
		if item['role'] == 'user' and idx < split_index:
			new_messages.append({'role': 'user', 'content': [{'type': 'text', 'text': f'screenshot {image_cnt}'}]})
			image_cnt += 1
		else:
			new_messages.append(item)
	return new_messages

def check_parameters(action, valid_parameters):
	for k, v in action.items():
		if k == 'action_type':
			continue
		elif k not in valid_parameters:
			assert v is None
		else:
			assert v is not None

def scale_to_qwen3_vl(coord, w_or_h):
	if coord > w_or_h:
		print(coord, w_or_h)
		coord = w_or_h
	new_coord = int(coord / w_or_h * 1000)
	assert 0 <= new_coord <= 1000
	return new_coord

def unify_action(ds_name, action):
	if ds_name == 'android_control':
		action_to_valid_parameters = {
			'click': ['x', 'y'],
			'long_press': ['x', 'y'],
			'scroll': ['direction'],
			'input_text': ['text'],
			'navigate_home': [],
			'navigate_back': [],
			'open_app': ['app_name'],
			'wait': [],
			'finish': []
		}
		valid_parameters = action_to_valid_parameters[action['action_type']]
		check_parameters(action, valid_parameters)
		unified_action = {'name': action['action_type'], "arguments": {}}
		for p in valid_parameters:
			if p == 'x':
				unified_action['arguments'][p] = scale_to_qwen3_vl(action[p], 1080)
			elif p == 'y':
				unified_action['arguments'][p] = scale_to_qwen3_vl(action[p], 2400)
			else:
				unified_action['arguments'][p] = action[p]
	return json.dumps(unified_action, ensure_ascii=False)

def save_b64_image(base64_string, save_path):
	if os.path.exists(save_path):
		return 
	image_data = base64.b64decode(base64_string)
	with open(save_path, "wb") as f:
		f.write(image_data)

def convert_android_control_to_multiturn(df, images_in_episode):
	all_episode_id = df['episode_id'].tolist()
	all_goal = df['goal'].tolist()
	all_screenshots_b64 = df['screenshots_b64'].tolist()
	all_actions = df['actions'].tolist()
	all_step_instructions = df['step_instructions'].tolist()

	data = []
	system = get_system('android_control')
	for episode_id, goal, screenshots_b64, actions, step_instructions in zip(all_episode_id, all_goal, all_screenshots_b64, all_actions, all_step_instructions):
		messages = [{'role': 'system', 'content': [{'type': 'text', 'text': system + goal}]}]
		os.makedirs(f'images/{episode_id}', exist_ok=True)
		actions = np.append(actions, {'action_type': 'finish'})
		step_instructions = np.append(step_instructions, 'Task completed.')
		assert len(screenshots_b64) == len(actions) == len(step_instructions), print(len(screenshots_b64), actions, step_instructions)
		for i, (b64, action, step_ins) in enumerate(zip(screenshots_b64, actions, step_instructions)):
			image_path = f'images/{episode_id}/screenshot-{i}.png'
			action_w_cot = '<think>' + step_ins + '</think>\n' + '<tool_call>' + unify_action('android_control', action) + '</tool_call>'
			if len(messages) == 1:
				messages.append({'role': 'user', 'content': [{'type': 'image', 'image': image_path}]})
				messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': action_w_cot}]})	
			else:
				messages = process_history(messages, images_in_episode-1)
				messages.append({'role': 'user', 'content': [{'type': 'image', 'image': image_path}]})
				messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': action_w_cot}]})
			data.append({'messages': messages, 'episode_id': episode_id})
			save_b64_image(b64, image_path)
	return data

if __name__ == '__main__':
	train, test = [], []
	data_dir = 'android-control'
	for f in os.listdir(data_dir):
		df = pd.read_parquet(os.path.join(data_dir, f))
		if 'train-' in f:
			train += convert_android_control_to_multiturn(df, 1)
		else:
			test += convert_android_control_to_multiturn(df, 1)
	print(len(train), len(test))
	to_json(train, 'data/train.txt')
	to_json(test, 'data/test.txt')