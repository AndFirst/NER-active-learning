from kivy.uix.button import Button
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView
from kivy.properties import StringProperty
from kivy.utils import platform
import os
from kivy.app import App
from kivy.uix.label import Label

kv_string = """
<ExistingProjectScreen>:
    BoxLayout:
        orientation: 'vertical'
        Button:
            text: 'Wybierz folder'
            size_hint: None, None
            size: 150, 50
            on_press: root.open_file_chooser()
        Label:
            text: 'Wybrana ścieżka: ' + root.selected_path
            color: 0, 0, 0, 1  # Kolor tekstu: czarny
        PrevNextButtons:
            id: prev_next_buttons
            back_text: "Wstecz"
            next_text: "Dalej"
"""

Builder.load_string(kv_string)


class ExistingProjectScreen(Screen):
    selected_path = StringProperty('')
    popup = None

    def __init__(self, **kwargs):
        super(ExistingProjectScreen, self).__init__(**kwargs)
        self.ids.prev_next_buttons.on_back = self.go_to_welcome
        self.ids.prev_next_buttons.on_next = self.check_files

    def go_to_welcome(self):
        self.manager.current = 'welcome'

    def go_to_main_menu(self):
        self.manager.current = 'main_menu'

    def open_file_chooser(self):
        app_path = App.get_running_app().home_dir
        if platform == 'win':
            filters = ['*']
        else:
            filters = [lambda folder, filename: os.path.isdir(
                os.path.join(folder, filename))]

        file_chooser = FileChooserIconView(
            path=app_path, filters=filters, dirselect=True)
        file_chooser.bind(on_submit=self.on_submit)

        # Dodanie przycisku "Wybierz" do okna dialogowego
        choose_button = Button(
            text='Wybierz', size_hint=(None, None), size=(150, 50))
        choose_button.bind(on_press=lambda instance: self.on_submit(
            file_chooser, file_chooser.selection, None))
        file_chooser.add_widget(choose_button)

        self.popup = Popup(title='Wybierz folder',
                           content=file_chooser, size_hint=(0.9, 0.9))
        self.popup.open()

    def on_submit(self, instance, selection, touch):
        if selection:
            self.selected_path = selection[0]
            print("Selected path:", self.selected_path)
        self.popup.dismiss()  # Zamknięcie okna dialogowego po wybraniu folderu

    def check_files(self):
        if self.selected_path:
            files = ['1.txt', '2.txt']
            missing_files = [f for f in files if not os.path.exists(
                os.path.join(self.selected_path, f))]
            if missing_files:
                Popup(title='Błąd', content=Label(text=f'Brak plików: {", ".join(missing_files)}'),
                      size_hint=(None, None), size=(400, 200)).open()
            else:
                print("Wszystkie wymagane pliki znalezione.")
                self.manager.current = 'main_menu'
        else:
            Popup(title='Błąd', content=Label(text='Nie wybrano ścieżki.'),
                  size_hint=(None, None), size=(300, 200)).open()
