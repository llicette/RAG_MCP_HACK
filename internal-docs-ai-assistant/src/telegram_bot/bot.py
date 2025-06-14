import asyncio
import os
import generator

from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    CallbackQuery,
    BotCommand
)
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram import Router
from aiogram.filters import Command
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

FILES_DIR = os.getenv("FILES_DIR")
BOT_TOKEN = os.getenv("BOT_TOKEN")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
router = Router()

commands = [BotCommand(command='start',
                           description='начать работу с ботом'),
                BotCommand(command='help',
                           description='помощь'),
                BotCommand(command='files',
                           description='все файлы'),
                BotCommand(command='delete',
                           description='удалить файл'),
                BotCommand(command='generate',
                           description='генерация Q&A')]

class DeleteFile(StatesGroup):
    waiting_for_file_selection = State()


@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer("👋 Добро пожаловать! "
                         "Я — ваш AI-ассистент по внутренней документации компании.  "
                         "Загружаю, анализирую и храню важные внутренние материалы, "
                         "чтобы быстро находить нужную информацию."
                         "\n\nОтправьте мне pdf-файл или задайте вопрос — я помогу разобраться!"
                         "\nОтправьте /help чтобы ознакомиться со списком всех доступных команд")


@dp.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer("✨ Что я умею:"
                         "\n/help - список всех команд"
                         "\n/files - список всех загруженных файлов"
                         "\n/delete - удаление файла"
                         "\n/generate - генерация вопросов и ответов по загруженным файлам"
                         "\nДля того, чтобы добавить файл в загруженные, "
                         "просто отправьте мне файл")


@dp.message(F.document)
async def handle_document(message: Message):
    document = message.document

    if not document.file_name.lower().endswith('.pdf'):
        await message.reply("❌ Я принимаю только PDF-файлы. Пожалуйста, загрузите документ в формате .pdf")
        return

    file_name = document.file_name
    user_id = message.from_user.id

    file_id = document.file_id
    file = await bot.get_file(file_id)

    dir = str(user_id)
    if not dir in os.listdir(FILES_DIR):
        os.makedirs(os.path.join(FILES_DIR, dir), exist_ok=True)

    file_path = os.path.join(FILES_DIR, dir, file_name)

    await bot.download_file(file.file_path, file_path)

    await message.answer(f"Файл '{file_name}' успешно загружен!")


@dp.message(Command("files"))
async def cmd_files(message: Message):
    user_id = message.from_user.id
    dir = str(user_id)

    if not dir in os.listdir(FILES_DIR):
        await message.answer("Вы ещё не загрузили ни одного файла.")
        return

    files = os.listdir(os.path.join(FILES_DIR, dir))
    file_list = '\n'.join(['• ' + os.path.basename(file) for file in files])
    await message.answer(f'Ваши загруженные файлы:\n{file_list}')


@dp.message(Command("delete"))
async def cmd_delete(message: Message):
    user_id = message.from_user.id
    dir = str(user_id)

    if not dir in os.listdir(FILES_DIR):
        await message.answer("Вы ещё не загрузили ни одного файла.")
        return

    files = os.listdir(os.path.join(FILES_DIR, dir))

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=os.path.basename(file),
                              callback_data=f"delete_{os.path.basename(file)}")]
        for file in files
    ])

    await message.answer("Нажмите на имя файла, чтобы удалить его:", reply_markup=keyboard)


@router.callback_query(lambda c: c.data.startswith('delete_'))
async def process_file_deletion(callback_query: CallbackQuery, state: FSMContext):
    selected_file = callback_query.data.replace('delete_', '', 1)

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Да", callback_data=f"confirm_{selected_file}"),
            InlineKeyboardButton(text="❌ Нет", callback_data="cancel_delete")
        ]
    ])

    await callback_query.message.edit_text(
        f"Вы уверены, что хотите удалить файл '{selected_file}'?",
        reply_markup=keyboard
    )
    await state.set_state(DeleteFile.waiting_for_file_selection)


@router.callback_query(lambda c: c.data.startswith('confirm_'))
async def confirm_deletion(callback_query: CallbackQuery, state: FSMContext):
    user_id = callback_query.from_user.id
    dir = str(user_id)

    file_to_delete = callback_query.data.replace('confirm_', '', 1)

    file_path = os.path.join(FILES_DIR, dir, file_to_delete)
    os.remove(file_path)

    await callback_query.message.edit_text(f"Файл '{file_to_delete}' успешно удалён.")
    await state.clear()


@router.callback_query(lambda c: c.data == "cancel_delete")
async def cancel_deletion(callback_query: CallbackQuery, state: FSMContext):
    await callback_query.message.edit_text("Удаление отменено.")
    await state.clear()


@router.message(Command("generate"))
async def cmd_generate(message: Message):
    user_id = message.from_user.id
    dir = str(user_id)

    if not dir in os.listdir(FILES_DIR):
        await message.answer("У вас нет загруженных файлов для анализа.")
        return

    await message.answer("🧠 Генерирую вопросы и ответы... Это может занять несколько секунд.")

    paths = [os.path.join(FILES_DIR, dir, file) for file in os.listdir(os.path.join(FILES_DIR, dir))]
    result = generator.generate(paths)
    for obj in result:
        await message.answer(f'Вопрос: {obj["question"]}\nОтвет: {obj["answer"]}')


dp.include_router(router)


async def main():
    print("Бот запущен...")
    await bot.set_my_commands(commands)
    await dp.start_polling(bot)


if __name__ == "__main__":
    os.makedirs(FILES_DIR, exist_ok=True)
    asyncio.run(main())
