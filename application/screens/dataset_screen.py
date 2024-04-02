from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView
from kivy.properties import StringProperty
from kivy.utils import platform
import os
from kivy.app import App
from kivy.uix.label import Label
from components.prev_next_buttons import PrevNextButtons
from kivy.uix.button import Button

kv_string = """
<DatasetScreen>:
    BoxLayout:
        orientation: 'vertical'
        Button:
            text: 'Wybierz plik'
            size_hint: None, None
            size: 150, 50
            on_press: root.open_file_chooser()
        Label:
            text: 'Wybrany plik: ' + root.selected_file
            color: 0, 0, 0, 1  # Kolor tekstu: czarny
        PrevNextButtons:
            id: prev_next_buttons
            back_text: "Wstecz"
            next_text: "Dalej"
"""

Builder.load_string(kv_string)


class DatasetScreen(Screen):
    selected_file = StringProperty('')  # Zmienna przechowująca wybrany plik
    popup = None  # Atrybut klasy do przechowywania referencji do okna dialogowego

    def __init__(self, **kwargs):
        shared_data = kwargs.pop('shared_data', None)
        super(DatasetScreen, self).__init__(**kwargs)
        self.shared_data = shared_data  # Przypisz shared_data do atrybutu klasy

        self.ids.prev_next_buttons.on_back = self.go_to_create_project
        self.ids.prev_next_buttons.on_next = self.check_file

    def go_to_create_project(self):
        self.manager.current = 'create_project'

    def go_to_add_labels(self):
        self.manager.current = 'add_labels'

    def open_file_chooser(self):
        app_path = App.get_running_app().home_dir
        if platform == 'win':
            filters = ['*.csv', '*.json', '*.jsonl']
        else:
            filters = [lambda folder, filename: any(
                filename.endswith(ext) for ext in ('.csv', '.json', '.jsonl'))]

        file_chooser = FileChooserIconView(
            path=app_path, filters=filters, dirselect=False)
        file_chooser.bind(on_submit=self.on_submit)

        # Dodanie przycisku "Wybierz" do okna dialogowego
        choose_button = Button(
            text='Wybierz', size_hint=(None, None), size=(150, 50))
        choose_button.bind(on_press=lambda instance: self.on_submit(
            file_chooser, file_chooser.selection, None))
        file_chooser.add_widget(choose_button)

        self.popup = Popup(title='Wybierz plik',
                           content=file_chooser, size_hint=(0.9, 0.9))
        self.popup.open()

    def on_submit(self, instance, selection, touch):
        if selection:
            self.selected_file = selection[0]
            print("Selected file:", self.selected_file)
        self.popup.dismiss()  # Zamknięcie okna dialogowego po wybraniu pliku

    def check_file(self):
        if self.selected_file:
            if os.path.exists(self.selected_file):
                print("Plik wybrany pomyślnie:", self.selected_file)
                self.shared_data.set_data('dataset_path', self.selected_file)
                print(self.shared_data.data)
                self.manager.current = 'add_labels'
            else:
                Popup(title='Błąd', content=Label(text='Wybrany plik nie istnieje.'),
                      size_hint=(None, None), size=(300, 200)).open()
        else:
            Popup(title='Błąd', content=Label(text='Nie wybrano pliku.'),
                  size_hint=(None, None), size=(300, 200)).open()
