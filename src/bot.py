import logging
import os
import subprocess
import re
import pandas as pd

from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv('API_TOKEN')
hostname = os.getenv('HOSTNAME')
data_path = os.getenv('DATA_PATH')
data = pd.read_csv(data_path + 'important copy.csv')

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply("Hi!\nI'm QA Bot for NUP!")


@dp.message_handler(commands=['ping'])
async def send_pong(message: types.Message):
    pong = ""
    try:
        result = subprocess.run(['ping', '-c', '5', hostname], capture_output=True, text=True)
        output = result.stdout
        regex = r"time=(\d+(\.\d+)?)"
        match = re.search(regex, output)
        if match:
            response_time = match.group(1)
            pong = f"Bot is up! Response time: {response_time} ms"
        else:
            pong = f"Bot is up! Unable to retrieve response time"
    except subprocess.CalledProcessError:
        pong = f"Bot is down!"

    await message.reply(pong)

@dp.message_handler(commands=['ask'])
async def echo(message: types.Message):
    user_question = message.text[len('/ask'):].strip()

    if user_question == '':
        await message.reply('No question provided!')
    else:
        await message.reply(user_question)

@dp.message_handler()
async def process_message(message: types.Message):
    if '#важное' in message.text:
        print(message.author_signature)
        # data.loc[len(df.index)] = [message.from_user.name, message.text]
        await message.reply('Added to data!')


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
