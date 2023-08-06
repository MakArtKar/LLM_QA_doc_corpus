import logging
import os
import subprocess
import re
import csv

from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv
from models.build_model import get_model

from models.model_refine import ModelRefine
from database import Database
from utils.utils import get_config

load_dotenv()
API_TOKEN = os.getenv('API_TOKEN')
hostname = os.getenv('HOSTNAME')
data_path = os.getenv('DATA_PATH')
config_name = os.getenv('CONFIG')
csv_path = os.path.join(data_path, 'important.csv')

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply("Hi!\nI'm QA Bot for NUP!")


@dp.message_handler(commands=['ping'])
async def send_pong(message: types.Message):
    try:
        pong = "Bot is up!"
    except subprocess.CalledProcessError:
        pong = "Bot is down!"

    await message.reply(pong)


@dp.message_handler(commands=['ask'])
async def echo(message: types.Message):
    user_question = message.text[len('/ask'):].strip()

    if user_question == '':
        await message.reply('No question provided!')
    else:
        await message.reply(model.response(user_question))


@dp.message_handler(commands=['add'])
async def process_message(message: types.Message):
    if message.reply_to_message:
        text = message.reply_to_message.text
    else:
        text = message.text[len("/add"):].strip()
    new_row = {'Name': message.from_user.first_name, 'Text': text}
    with open(data_path + 'important copy.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_row.keys())
        writer.writerow(new_row)
    await message.reply('Added to data!')


if __name__ == '__main__':
    config = get_config(config_name)
    model = get_model(
        Database(config['sentence_transformer']), 
        config['strategy'], 
        config['model_id'], 
        config['task'], 
        config['model_kwargs'],
        config['prompts'],
    )
    executor.start_polling(dp, skip_updates=True)
