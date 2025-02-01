import cv2
import tkinter as tk
from tkinter import ttk, messagebox, Toplevel, filedialog
from PIL import Image, ImageTk
import os
import sys
import time
import threading
import sqlite3
import json
from math import ceil

# Para detectar las pantallas conectadas (requerido: pip install screeninfo)
try:
    from screeninfo import get_monitors
except ImportError:
    print("No se encontró el módulo screeninfo. Instálalo con: pip install screeninfo")
    get_monitors = None

############################################
# MÓDULO DE BASE DE DATOS
############################################
def get_base_path():
    """
    Retorna la ruta base para buscar archivos (DB, imágenes).
    Si se está ejecutando con PyInstaller (sys.frozen=True),
    utiliza la ruta del ejecutable. De lo contrario, usa la del script .py.
    """
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return base_dir

def connect_db():
    """
    Conecta a la base de datos SQLite y asegura que la tabla 'cameras' exista.
    Retorna la conexión y el cursor.
    """
    base_dir = get_base_path()
    db_path = os.path.join(base_dir, 'camaras.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            url TEXT NOT NULL
        )
    ''')
    conn.commit()
    return conn, cursor

def load_cameras(cursor):
    """
    Carga todas las cámaras de la base de datos.
    Retorna una lista de tuplas: [(id, name, url), ...]
    """
    try:
        cursor.execute("SELECT id, name, url FROM cameras")
        return cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"Error al cargar cámaras: {e}")
        return []

def add_camera_db(cursor, conn, name, url):
    cursor.execute("INSERT INTO cameras (name, url) VALUES (?, ?)", (name, url))
    conn.commit()
    return cursor.lastrowid

def remove_camera_db(cursor, conn, camera_id):
    cursor.execute("DELETE FROM cameras WHERE id=?", (camera_id,))
    conn.commit()

def edit_camera_db(cursor, conn, camera_id, new_name, new_url):
    cursor.execute("UPDATE cameras SET name=?, url=? WHERE id=?", (new_name, new_url, camera_id))
    conn.commit()

############################################
# CLASE VideoStream PARA STREAMING
############################################
class VideoStream:
    """
    Hilo independiente para leer frames de la cámara en segundo plano.
    Incluye un contador de fallos consecutivos (fail_count) para reconectar
    automáticamente cuando la cámara deja de enviar frames (por ejemplo, tras
    un reinicio físico o un corte de red).
    """
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_ANY)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()

        if not self.cap.isOpened():
            print(f"No se pudo abrir la cámara: {self.src}")
            self.ret = False

        # --- Contador de fallos para reconectar ---
        self.fail_count = 0
        self.max_fail = 100  # Umbral de reintentos (lecturas fallidas consecutivas)

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Resetear contador de fallos cuando se lee bien
                with self.lock:
                    self.ret = True
                    self.frame = frame
                self.fail_count = 0
            else:
                # Contabilizar el fallo
                self.ret = False
                self.fail_count += 1

                # Si supera el umbral, intentar reconectar
                if self.fail_count > self.max_fail:
                    print(f"[RECONEXIÓN] Reiniciando la cámara: {self.src} (fallos consecutivos={self.fail_count})")
                    self.cap.release()
                    time.sleep(2)  # Pausa para dar tiempo a la cámara tras reiniciarse
                    self.cap = cv2.VideoCapture(self.src, cv2.CAP_ANY)
                    self.fail_count = 0
                    if not self.cap.isOpened():
                        print(f"[RECONEXIÓN] No se pudo reabrir la cámara: {self.src}")
                    else:
                        print(f"[RECONEXIÓN] Cámara {self.src} reabierta correctamente.")

            time.sleep(0.01)  # Evitar uso excesivo de CPU

    def get_frame(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
            else:
                return False, None

    def stop(self):
        self.running = False
        self.cap.release()

############################################
# FUNCIÓN AUXILIAR PARA AJUSTE "FILL" (SIN DEFORMAR)
############################################
def resize_and_crop_to_fill(frame, target_width, target_height):
    """
    Redimensiona la imagen 'frame' para que LLENE completamente
    el espacio (target_width x target_height) SIN dejar barras negras
    y SIN deformar, recortando lo que sobre.
    """
    h, w = frame.shape[:2]
    scale_w = target_width / w
    scale_h = target_height / h
    scale = max(scale_w, scale_h)  # Escala para llenar
    new_w = int(w * scale)
    new_h = int(h * scale)
    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Recortar el centro si sobra
    x_center = new_w // 2
    y_center = new_h // 2
    half_tw = target_width // 2
    half_th = target_height // 2

    left = x_center - half_tw
    right = left + target_width
    top = y_center - half_th
    bottom = top + target_height

    if left < 0:
        left = 0
        right = target_width
    if top < 0:
        top = 0
        bottom = target_height
    if right > new_w:
        right = new_w
        left = new_w - target_width
    if bottom > new_h:
        bottom = new_h
        top = new_h - target_height

    frame = frame[top:bottom, left:right]
    return frame

############################################
# CLASE FullscreenWindow
############################################
class FullscreenWindow:
    """
    Ventana secundaria para mostrar el streaming en pantalla completa sin controles superiores.
    Se muestran los streams en dos modos: "Single" y "Grid4".
    En "Grid4", permite hasta 20 cámaras, mostrando 4 a la vez y rotando cada rotation_time segundos.
    En "Single", permite mostrar múltiples cámaras rotando entre ellas cada rotation_time segundos.
    """
    def __init__(self, parent, mode, cameras, streams,
                 frame_rate=30, auto_rotate=True, rotation_time=10, monitor_geom=None):
        self.parent = parent
        self.mode = mode
        self.all_cameras = cameras   # [(id, name, url), ...]
        self.streams = streams       # Diccionario de VideoStream
        self.frame_rate = frame_rate
        self.auto_rotate = auto_rotate
        self.rotation_time = rotation_time

        # Variables de navegación para Grid4
        self.grid_size = 4 if mode == "Grid4" else None
        self.max_grid_cameras = 20 if mode == "Grid4" else None
        self.current_grid_page = 0

        # Variables de navegación para Single
        self.single_camera_index = 0

        # Lista de cámaras asignadas para evitar duplicados en "Todas las Pantallas"
        self.assigned_cameras = set()

        # Crear la ventana Toplevel
        self.root = Toplevel(parent.root)
        self.root.title(f"Streaming - {self.mode}")
        self.root.configure(bg="#222222")

        # Si se pasó la geometría de monitor, posicionar la ventana en esa pantalla
        if monitor_geom:
            geom = f"{monitor_geom['width']}x{monitor_geom['height']}+{monitor_geom['x']}+{monitor_geom['y']}"
            self.root.geometry(geom)
            self.root.update_idletasks()  # Asegurar que la geometría se aplique
            self.root.lift()  # Traer la ventana al frente
            self.root.attributes("-topmost", True)  # Mantener la ventana al frente
            self.root.attributes("-fullscreen", True)  # Establecer en fullscreen
        else:
            self.root.geometry("900x650")
            try:
                self.root.state('zoomed')  # Windows
            except:
                self.root.attributes("-zoomed", True)  # Otros sistemas

        # Marco superior para botones (header)
        self.button_frame = ttk.Frame(self.root)
        # Colocarlos arriba sin padding para que aparezcan tipo header
        self.button_frame.pack(side="top", fill="x", padx=0, pady=0)

        # Botón para cerrar
        self.close_button = ttk.Button(self.button_frame, text="Cerrar Ventana", command=self.close_window)
        self.close_button.pack(side="left", padx=0, pady=0)

        # Botón para salir de fullscreen (opcional)
        self.exit_fullscreen_button = ttk.Button(self.button_frame, text="Salir de Pantalla Completa", command=self.exit_fullscreen)
        self.exit_fullscreen_button.pack(side="right", padx=0, pady=0)

        # Área principal de streaming
        self.body = ttk.Frame(self.root, style="Main.TFrame")
        self.body.pack(fill="both", expand=True)

        self.rotation_job = None

        # Vincular ESC para salir de fullscreen
        self.root.bind("<Escape>", self.exit_fullscreen)

        # Construir vista según modo
        if self.mode == "Single":
            self.build_single_view()
        elif self.mode == "Grid4":
            self.build_grid_view()

        # Iniciar rotación automática si está activada
        if self.auto_rotate:
            self.start_auto_rotation()

        # Vincular el cierre de la ventana para detener rotaciones
        self.root.protocol("WM_DELETE_WINDOW", self.close_window)

    ################################################
    # Manejo de pantalla completa
    ################################################
    def enter_fullscreen(self):
        """
        Ajusta la ventana a la pantalla elegida en el combobox y entra en fullscreen.
        """
        monitor_name = self.selected_monitor.get()
        if monitor_name == "Todas las Pantallas":
            # No se aplica fullscreen a nivel de ventana individual
            pass
        else:
            monitor_geom = self.parent.get_monitor_geometry(monitor_name)
            if monitor_geom:
                geom = f"{monitor_geom['width']}x{monitor_geom['height']}+{monitor_geom['x']}+{monitor_geom['y']}"
                self.root.geometry(geom)

            self.root.attributes("-fullscreen", True)
            # Ocultar el marco de botones
            self.button_frame.pack_forget()

    def exit_fullscreen(self, event=None):
        self.root.attributes("-fullscreen", False)
        # Volver a mostrar el marco de botones
        self.button_frame.pack(side="top", fill="x", padx=0, pady=0)

    ################################################
    # Construcción y actualización de "Single"
    ################################################
    def build_single_view(self):
        # Usar un tk.Label en lugar de ttk para que el bg sea #000000 real
        self.single_label = tk.Label(self.body, text="", bg="#000000")
        self.single_label.pack(fill="both", expand=True)
        self.update_single_live()

    def update_single_live(self):
        if not self.all_cameras:
            self.single_label.config(text="No hay cámaras", image="")
        else:
            if self.single_camera_index >= len(self.all_cameras):
                self.single_camera_index = 0  # Resetear índice si excede

            cid, name, url = self.all_cameras[self.single_camera_index]
            vs = self.streams.get(cid)
            if vs:
                ret, frame = vs.get_frame()
                if ret and frame is not None:
                    try:
                        self.single_label.update_idletasks()
                        w = self.single_label.winfo_width() or 800
                        h = self.single_label.winfo_height() or 600

                        if w > 1 and h > 1:
                            frame = resize_and_crop_to_fill(frame, w, h)
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(frame_rgb)
                            imgtk = ImageTk.PhotoImage(image=img)
                            self.single_label.config(image=imgtk, text="")
                            self.single_label.image = imgtk
                        else:
                            self.single_label.config(text="Cargando...", image="")
                    except Exception as e:
                        self.single_label.config(text=f"Error: {e}", image="")
                else:
                    # Cámara fallida, reasignar
                    self.handle_camera_failure(cid)
            else:
                self.single_label.config(text="Cámara no encontrada", image="")
        delay = int(1000 / self.frame_rate)
        self.root.after(delay, self.update_single_live)

    ################################################
    # Construcción y actualización de "Grid4"
    ################################################
    def build_grid_view(self):
        # Frame que contendrá la cuadrícula 2x2
        self.grid_frame = tk.Frame(self.body, bg="black")
        self.grid_frame.pack(fill="both", expand=True)

        # Fuerza a que las 2 filas y 2 columnas tengan el mismo tamaño
        for i in range(2):
            self.grid_frame.rowconfigure(i, weight=1, uniform="row")
            self.grid_frame.columnconfigure(i, weight=1, uniform="col")

        self.labels = []
        self.build_grid_labels()
        self.update_grid_live()

    def build_grid_labels(self):
        # Elimina labels anteriores (si existían)
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        self.labels.clear()

        # Toma el subset de cámaras para mostrar (4 por página en Grid4)
        subset = self.get_current_grid_subset()

        for idx in range(self.grid_size):
            if idx < len(subset):
                cid, name, url = subset[idx]
                # Cada celda tendrá un Label con fondo negro
                lbl = tk.Label(self.grid_frame, text="", bg="#000000")
                # Asigna la posición en la cuadrícula y padding
                lbl.grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=10, pady=10)
                lbl.camera_id = cid  # Guardar el ID de la cámara asignada
                self.labels.append((cid, lbl))
                self.assigned_cameras.add(cid)
            else:
                # Celdas vacías si hay menos de 4 cámaras en el subset
                lbl = tk.Label(self.grid_frame, text="", bg="#111111")
                lbl.grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=10, pady=10)
                lbl.camera_id = None
                self.labels.append((None, lbl))

    def get_current_grid_subset(self):
        start = self.current_grid_page * self.grid_size
        end = start + self.grid_size
        return self.all_cameras[start:end]

    def update_grid_live(self):
        subset = self.get_current_grid_subset()
        for idx, (cid, lbl) in enumerate(self.labels):
            if idx < len(subset):
                cam_id, name, url = subset[idx]
                vs = self.streams.get(cam_id)
                if vs:
                    ret, frame = vs.get_frame()
                    if ret and frame is not None:
                        try:
                            # Obtén el tamaño actual del Label
                            label_width = lbl.winfo_width()
                            label_height = lbl.winfo_height()

                            if label_width > 1 and label_height > 1:
                                # Redimensionar y recortar para llenar 100%
                                frame = resize_and_crop_to_fill(frame, label_width, label_height)
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                img = Image.fromarray(frame_rgb)
                                imgtk = ImageTk.PhotoImage(image=img)
                                lbl.config(image=imgtk, text="")
                                lbl.image = imgtk
                            else:
                                lbl.config(text="Cargando...", image="")
                        except Exception:
                            lbl.config(text="Error", image="")
                    else:
                        # Cámara fallida, reasignar
                        self.handle_grid_camera_failure(cam_id, lbl)
                else:
                    lbl.config(text="Cámara no encontrada", image="")
            else:
                # Celdas vacías
                lbl.config(text="", image="")

        # Vuelve a actualizar en el siguiente frame
        delay = int(1000 / self.frame_rate)
        self.root.after(delay, self.update_grid_live)

    # --- Rotación Automática ---
    def start_auto_rotation(self):
        if self.mode == "Grid4":
            self.rotation_job = self.root.after(self.rotation_time * 1000, self.rotate_grid)
        elif self.mode == "Single":
            self.rotation_job = self.root.after(self.rotation_time * 1000, self.rotate_single)

    def rotate_grid(self):
        if self.mode != "Grid4":
            return
        self.next_grid_page()
        self.build_grid_labels()
        self.update_grid_live()
        if self.auto_rotate:
            self.rotation_job = self.root.after(self.rotation_time * 1000, self.rotate_grid)

    def rotate_single(self):
        if self.mode != "Single":
            return
        if not self.all_cameras:
            return
        self.single_camera_index = (self.single_camera_index + 1) % len(self.all_cameras)
        if self.auto_rotate:
            self.rotation_job = self.root.after(self.rotation_time * 1000, self.rotate_single)

    def next_grid_page(self):
        total_pages = ceil(len(self.all_cameras) / self.grid_size)
        if total_pages == 0:
            return
        self.current_grid_page = (self.current_grid_page + 1) % total_pages

    def close_window(self):
        if self.rotation_job:
            self.root.after_cancel(self.rotation_job)
        self.root.destroy()

    def update_frame_rate(self, new_fps):
        self.frame_rate = new_fps

    ################################################
    # Manejo de Fallas de Cámaras en "Single"
    ################################################
    def handle_camera_failure(self, failed_cid):
        """
        Maneja la falla de una cámara en modo 'Single'.
        Reasigna a otra cámara disponible.
        """
        print(f"Falla detectada en la cámara ID: {failed_cid}")
        self.assigned_cameras.discard(failed_cid)  # Remover de asignadas

        # Buscar la próxima cámara disponible que no esté asignada
        for idx in range(len(self.all_cameras)):
            next_index = (self.single_camera_index + 1 + idx) % len(self.all_cameras)
            next_cid, name, url = self.all_cameras[next_index]
            if next_cid not in self.assigned_cameras:
                self.single_camera_index = next_index
                self.assigned_cameras.add(next_cid)
                print(f"Reasignando a la cámara ID: {next_cid}")
                break
        else:
            # Si no hay cámaras disponibles, mantener la actual
            print("No hay cámaras disponibles para reasignar.")

    ################################################
    # Manejo de Fallas de Cámaras en "Grid4"
    ################################################
    def handle_grid_camera_failure(self, failed_cid, lbl):
        """
        Maneja la falla de una cámara en modo 'Grid4'.
        Reasigna a otra cámara disponible en el slot correspondiente.
        """
        print(f"Falla detectada en la cámara ID: {failed_cid} en modo Grid")
        self.assigned_cameras.discard(failed_cid)  # Remover de asignadas

        # Buscar la próxima cámara disponible que no esté asignada
        for cam in self.all_cameras:
            next_cid, name, url = cam
            if next_cid not in self.assigned_cameras:
                self.assigned_cameras.add(next_cid)
                vs = self.streams.get(next_cid)
                if vs:
                    ret, frame = vs.get_frame()
                    if ret and frame is not None:
                        try:
                            # Obtén el tamaño actual del Label
                            label_width = lbl.winfo_width()
                            label_height = lbl.winfo_height()

                            if label_width > 1 and label_height > 1:
                                # Redimensionar y recortar para llenar 100%
                                frame = resize_and_crop_to_fill(frame, label_width, label_height)
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                img = Image.fromarray(frame_rgb)
                                imgtk = ImageTk.PhotoImage(image=img)
                                lbl.config(image=imgtk, text="")
                                lbl.image = imgtk
                                lbl.camera_id = next_cid  # Asignar nuevo ID
                                print(f"Reasignando a la cámara ID: {next_cid} en Grid")
                            else:
                                lbl.config(text="Cargando...", image="")
                        except Exception:
                            lbl.config(text="Error", image="")
                    else:
                        lbl.config(text="Cargando...", image="")
                break
        else:
            # Si no hay cámaras disponibles, mostrar imagen de error o mantener la actual
            lbl.config(text="Cámara no disponible", image="")
            print("No hay cámaras disponibles para reasignar en Grid.")

    def __del__(self):
        pass

############################################
# CLASE CameraApp (Ventana Principal – Menú)
############################################
class CameraApp:
    """
    Clase principal que muestra un menú con opciones para gestionar
    las cámaras (Agregar, Editar, Importar, Eliminar) y para abrir las vistas en
    ventana nueva: Individual, Cuadrícula 4.
    Incluye una interfaz para seleccionar qué cámaras mostrar.

    Se agrega un nuevo botón "Actualizar" para refrescar la conexión/streaming de cada cámara.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Menú – Administrador de Cámaras")
        self.root.geometry("1000x800")
        self.root.minsize(800, 600)
        self.fullscreen = False

        self.root.resizable(True, True)

        # Estilos con ttk
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("Main.TFrame", background="#F0F0F0")
        self.style.configure("TButton", font=("Arial", 12), padding=10)
        self.style.configure("TLabel", font=("Arial", 12), padding=5)
        self.style.configure("Header.TLabel", font=("Arial", 24, "bold"), foreground="#333333", background="#F0F0F0")
        self.style.configure("SubHeader.TLabel", font=("Arial", 14, "bold"), foreground="#333333", background="#F0F0F0")
        self.style.configure("TEntry", font=("Arial", 12))

        # Conexión a la base de datos y carga de cámaras
        self.conn, self.cursor = connect_db()
        self.camera_list = load_cameras(self.cursor)
        self.camera_list_sorted = sorted(self.camera_list, key=lambda c: c[0])
        self.streams = {}
        self.frame_rate = 30  # FPS por defecto
        self.auto_rotate = True
        self.rotation_time = 10  # Segundos de rotación

        # Lista de ventanas de streaming abiertas
        self.streaming_windows = []

        # Inicializar thumbnails
        self.thumbnails = {}
        self.load_thumbnails()

        # Iniciar VideoStreams
        self.initialize_streams()

        # Construir la interfaz
        self.build_menu()

    def load_thumbnails(self):
        """
        Carga las miniaturas para todas las cámaras.
        """
        for cid, name, url in self.camera_list_sorted:
            thumb = self.get_camera_thumbnail(url)
            if thumb is not None:
                thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(thumb)
                img = img.resize((100, 75), Image.Resampling.LANCZOS)
            else:
                img = Image.new('RGB', (100, 75), color='gray')
            imgtk = ImageTk.PhotoImage(image=img)
            self.thumbnails[cid] = imgtk

    def initialize_streams(self):
        """
        Inicia los VideoStreams para cada cámara.
        """
        for cid, name, url in self.camera_list_sorted:
            vs = VideoStream(url)
            vs.start()
            self.streams[cid] = vs

    def build_menu(self):
        main_frame = ttk.Frame(self.root, style="Main.TFrame")
        main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Título
        title = ttk.Label(main_frame, text="Administrador de Cámaras", style="Header.TLabel")
        title.grid(row=0, column=0, pady=(0, 20), sticky="n")

        # Lista de cámaras
        cameras_frame = ttk.Frame(main_frame, style="Main.TFrame")
        cameras_frame.grid(row=1, column=0, sticky="nsew")

        cameras_frame.grid_rowconfigure(0, weight=1)
        cameras_frame.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(cameras_frame, bg="#F0F0F0")
        scrollbar = ttk.Scrollbar(cameras_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas, style="Main.TFrame")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        cameras_frame.grid_rowconfigure(0, weight=1)
        cameras_frame.grid_columnconfigure(0, weight=1)

        for cid, name, url in self.camera_list_sorted:
            self.add_camera_to_list(self.scrollable_frame, cid, name, url)

        # Botones para abrir vistas de streaming
        streaming_frame = ttk.Frame(main_frame, style="Main.TFrame")
        streaming_frame.grid(row=2, column=0, pady=20, sticky="ew")

        streaming_frame.grid_columnconfigure(0, weight=1)
        streaming_frame.grid_columnconfigure(1, weight=1)

        btn_individual = ttk.Button(streaming_frame, text="Ver Cámaras Individuales",
                                    command=self.select_cameras_single)
        btn_individual.grid(row=0, column=0, padx=5, pady=10, sticky="ew")

        btn_grid4 = ttk.Button(streaming_frame, text="Ver Cuadrícula 4",
                               command=lambda: self.select_cameras_grid(4))
        btn_grid4.grid(row=0, column=1, padx=5, pady=10, sticky="ew")

        # Separador
        separator = ttk.Separator(main_frame, orient='horizontal')
        separator.grid(row=3, column=0, sticky="ew", pady=20)

        # Opciones de gestión de cámaras
        gestion_frame = ttk.Frame(main_frame, style="Main.TFrame")
        gestion_frame.grid(row=4, column=0, pady=10, sticky="ew")

        gestion_frame.grid_columnconfigure(0, weight=1)
        gestion_frame.grid_columnconfigure(1, weight=1)
        gestion_frame.grid_columnconfigure(2, weight=1)  # Nueva columna para el botón de reinicio

        btn_add = ttk.Button(gestion_frame, text="Agregar Cámara", command=self.prompt_add_camera)
        btn_add.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        btn_import = ttk.Button(gestion_frame, text="Importar JSON", command=self.import_cameras_json)
        btn_import.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        # Nuevo botón para reiniciar todas las cámaras
        btn_restart_all = ttk.Button(gestion_frame, text="Reiniciar Todas las Cámaras", command=self.restart_all_cameras)
        btn_restart_all.grid(row=0, column=2, padx=10, pady=5, sticky="ew")

        # Separador
        separator2 = ttk.Separator(main_frame, orient='horizontal')
        separator2.grid(row=5, column=0, sticky="ew", pady=20)

        # Configuración adicional
        config_frame = ttk.Frame(main_frame, style="Main.TFrame")
        config_frame.grid(row=6, column=0, pady=10, sticky="ew")

        config_frame.grid_columnconfigure(0, weight=1)
        config_frame.grid_columnconfigure(1, weight=1)
        config_frame.grid_columnconfigure(2, weight=1)

        # Velocidad de Frame (FPS)
        lbl_fps = ttk.Label(config_frame, text="Velocidad de Frame (FPS):", style="SubHeader.TLabel")
        lbl_fps.grid(row=0, column=0, padx=10, pady=5, sticky="e")

        self.frame_rate_var = tk.IntVar(value=self.frame_rate)
        self.frame_rate_combobox = ttk.Combobox(config_frame, textvariable=self.frame_rate_var,
                                                values=[1, 3, 5, 10, 15, 20, 30, 60],
                                                state="readonly")
        self.frame_rate_combobox.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.frame_rate_combobox.set(self.frame_rate)

        btn_apply_fps = ttk.Button(config_frame, text="Aplicar FPS", command=self.apply_frame_rate)
        btn_apply_fps.grid(row=0, column=2, padx=10, pady=5, sticky="w")

        # Tiempo de Rotación
        lbl_rot = ttk.Label(config_frame, text="Tiempo de Rotación (s):", style="SubHeader.TLabel")
        lbl_rot.grid(row=1, column=0, padx=10, pady=5, sticky="e")

        self.rotation_entry = ttk.Entry(config_frame, width=10)
        self.rotation_entry.insert(0, str(self.rotation_time))
        self.rotation_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        btn_aplicar_rot = ttk.Button(config_frame, text="Aplicar", command=self.apply_rotation_time)
        btn_aplicar_rot.grid(row=1, column=2, padx=10, pady=5, sticky="w")

        # Botón para alternar pantalla completa (ventana principal)
        btn_fullscreen = ttk.Button(config_frame, text="Pantalla Completa", command=self.toggle_fullscreen)
        btn_fullscreen.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        # Selección de Pantalla de Salida
        pant_frame = ttk.Frame(main_frame, style="Main.TFrame")
        pant_frame.grid(row=7, column=0, pady=10, sticky="ew")

        pant_frame.grid_columnconfigure(0, weight=1)
        pant_frame.grid_columnconfigure(1, weight=2)

        lbl_pantalla = ttk.Label(pant_frame, text="Selecciona Pantalla de Salida:", style="SubHeader.TLabel")
        lbl_pantalla.grid(row=0, column=0, padx=10, pady=5, sticky="e")

        self.selected_monitor = tk.StringVar(value="Principal")
        monitors = self.get_monitors_list()
        self.option_monitor = ttk.Combobox(pant_frame, textvariable=self.selected_monitor,
                                           values=monitors, state="readonly")
        self.option_monitor.grid(row=0, column=1, padx=10, pady=5, sticky="w")

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def add_camera_to_list(self, parent, cid, name, url):
        camera_frame = ttk.Frame(parent, style="Main.TFrame", borderwidth=1, relief="solid")
        camera_frame.grid(sticky="ew", pady=5, padx=10)
        camera_frame.grid_columnconfigure(1, weight=1)

        # Thumbnail
        if cid in self.thumbnails:
            imgtk = self.thumbnails[cid]
        else:
            img = Image.new('RGB', (100, 75), color='gray')
            imgtk = ImageTk.PhotoImage(image=img)
            self.thumbnails[cid] = imgtk

        lbl_thumb = ttk.Label(camera_frame, image=imgtk)
        lbl_thumb.image = imgtk
        lbl_thumb.grid(row=0, column=0, rowspan=3, padx=5, pady=5, sticky="w")

        # Información de la cámara
        info_frame = ttk.Frame(camera_frame, style="Main.TFrame")
        info_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=5)
        info_frame.grid_rowconfigure(0, weight=1)
        info_frame.grid_columnconfigure(0, weight=1)

        lbl_name = ttk.Label(info_frame, text=f"Nombre: {name}", style="SubHeader.TLabel")
        lbl_name.grid(row=0, column=0, sticky="w")

        lbl_url = ttk.Label(info_frame, text=f"URL: {url}", wraplength=600)
        lbl_url.grid(row=1, column=0, sticky="w")

        # Botones de editar, actualizar y eliminar
        btn_frame = ttk.Frame(camera_frame, style="Main.TFrame")
        btn_frame.grid(row=0, column=2, rowspan=3, padx=10, pady=5, sticky="e")

        btn_edit = ttk.Button(btn_frame, text="Editar", command=lambda c_id=cid: self.edit_camera_window(c_id))
        btn_edit.pack(pady=2, fill="x")

        # Nuevo botón "Actualizar" para refrescar la cámara
        btn_refresh = ttk.Button(btn_frame, text="Actualizar", command=lambda c_id=cid: self.refresh_camera_connection(c_id))
        btn_refresh.pack(pady=2, fill="x")

        btn_del = ttk.Button(btn_frame, text="Eliminar", command=lambda c_id=cid: self.delete_camera(c_id))
        btn_del.pack(pady=2, fill="x")

    def refresh_camera_connection(self, camera_id):
        """
        Vuelve a leer la URL desde la base de datos y reinicia el streaming
        (para refrescar la IP/conexión de la cámara).
        También actualiza la miniatura en caso de que haya cambiado.
        """
        try:
            self.cursor.execute("SELECT name, url FROM cameras WHERE id=?", (camera_id,))
            row = self.cursor.fetchone()
            if not row:
                messagebox.showerror("Error", "Cámara no encontrada en la base de datos.")
                return

            name, url = row[0], row[1]

            # Detener el stream anterior
            if camera_id in self.streams:
                self.streams[camera_id].stop()

            # Crear un nuevo stream
            new_vs = VideoStream(url)
            new_vs.start()
            self.streams[camera_id] = new_vs

            # Actualizar thumbnail
            thumb = self.get_camera_thumbnail(url)
            if thumb is not None:
                thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(thumb)
                img = img.resize((100, 75), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
            else:
                img = Image.new('RGB', (100, 75), color='gray')
                imgtk = ImageTk.PhotoImage(image=img)
            self.thumbnails[camera_id] = imgtk

            # Refrescar la lista de cámaras en la interfaz
            self.refresh_camera_list()
            messagebox.showinfo("Actualizado", f"Cámara '{name}' refrescada correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo actualizar la cámara: {e}")

    def get_camera_thumbnail(self, url):
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_ANY)
            ret, frame = cap.read()
            cap.release()
            if ret:
                return frame
            return None
        except:
            return None

    def edit_camera_window(self, camera_id):
        cam = next((c for c in self.camera_list if c[0] == camera_id), None)
        if not cam:
            messagebox.showerror("Error", "Cámara no encontrada.")
            return
        cid, current_name, current_url = cam

        ew = Toplevel(self.root)
        ew.title("Editar Cámara")
        ew.geometry("400x450")
        ew.resizable(False, False)
        ew.configure(bg="#F0F0F0")

        ew.grid_rowconfigure(0, weight=1)
        ew.grid_rowconfigure(1, weight=1)
        ew.grid_rowconfigure(2, weight=1)
        ew.grid_rowconfigure(3, weight=1)
        ew.grid_rowconfigure(4, weight=1)
        ew.grid_columnconfigure(0, weight=1)
        ew.grid_columnconfigure(1, weight=3)

        lbl_name = ttk.Label(ew, text="Nombre de la Cámara:", style="SubHeader.TLabel")
        lbl_name.grid(row=0, column=0, padx=10, pady=10, sticky="e")

        name_entry = ttk.Entry(ew, width=30)
        name_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        name_entry.insert(0, current_name)

        lbl_url = ttk.Label(ew, text="URL RTSP:", style="SubHeader.TLabel")
        lbl_url.grid(row=1, column=0, padx=10, pady=10, sticky="e")

        url_entry = ttk.Entry(ew, width=30)
        url_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        url_entry.insert(0, current_url)

        lbl_thumb = ttk.Label(ew, text="Miniatura:", style="SubHeader.TLabel")
        lbl_thumb.grid(row=2, column=0, padx=10, pady=10, sticky="e")

        thumb_label = ttk.Label(ew)
        thumb_label.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        thumb = self.get_camera_thumbnail(current_url)
        if thumb is not None:
            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(thumb)
            img = img.resize((100, 75), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            thumb_label.config(image=imgtk)
            thumb_label.image = imgtk
        else:
            placeholder = Image.new('RGB', (100, 75), color='gray')
            imgtk = ImageTk.PhotoImage(image=placeholder)
            thumb_label.config(image=imgtk)
            thumb_label.image = imgtk

        def update_thumb():
            new_url = url_entry.get().strip()
            thumb = self.get_camera_thumbnail(new_url)
            if thumb is not None:
                thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(thumb)
                img = img.resize((100, 75), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                thumb_label.config(image=imgtk)
                thumb_label.image = imgtk
                self.thumbnails[cid] = imgtk
                messagebox.showinfo("Éxito", "Miniatura actualizada.")
            else:
                messagebox.showerror("Error", "No se pudo obtener miniatura.")

        btn_update_thumb = ttk.Button(ew, text="Actualizar Miniatura", command=update_thumb)
        btn_update_thumb.grid(row=3, column=1, padx=10, pady=5, sticky="w")

        def save_changes():
            new_name = name_entry.get().strip()
            new_url = url_entry.get().strip()
            if not new_name or not new_url:
                messagebox.showerror("Error", "Complete todos los campos.")
                return
            edit_camera_db(self.cursor, self.conn, cid, new_name, new_url)
            for idx, cc in enumerate(self.camera_list):
                if cc[0] == cid:
                    self.camera_list[idx] = (cid, new_name, new_url)
                    break
            self.camera_list_sorted = sorted(self.camera_list, key=lambda c: c[0])

            thumb = self.get_camera_thumbnail(new_url)
            if thumb is not None:
                thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(thumb)
                img = img.resize((100, 75), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.thumbnails[cid] = imgtk
            else:
                img = Image.new('RGB', (100, 75), color='gray')
                imgtk = ImageTk.PhotoImage(image=img)
                self.thumbnails[cid] = imgtk

            if cid in self.streams:
                self.streams[cid].stop()
                vs = VideoStream(new_url)
                vs.start()
                self.streams[cid] = vs

            self.refresh_camera_list()
            messagebox.showinfo("Éxito", "Cámara editada.")
            ew.destroy()

        btn_save = ttk.Button(ew, text="Guardar Cambios", command=save_changes)
        btn_save.grid(row=4, column=0, columnspan=2, padx=10, pady=20, sticky="ew")

    def delete_camera(self, camera_id):
        confirm = messagebox.askyesno("Confirmar", "¿Eliminar la cámara?")
        if confirm:
            remove_camera_db(self.cursor, self.conn, camera_id)
            self.camera_list = [c for c in self.camera_list if c[0] != camera_id]
            self.camera_list_sorted = sorted(self.camera_list, key=lambda c: c[0])
            if camera_id in self.thumbnails:
                del self.thumbnails[camera_id]
            if camera_id in self.streams:
                self.streams[camera_id].stop()
                del self.streams[camera_id]
            self.refresh_camera_list()

            # Cerrar ventanas de streaming que incluyan esta cámara
            for window in self.streaming_windows[:]:
                if window.mode == "Single":
                    if any(cam[0] == camera_id for cam in window.all_cameras):
                        window.close_window()
                        self.streaming_windows.remove(window)
                elif window.mode == "Grid4":
                    if any(cam[0] == camera_id for cam in window.all_cameras):
                        window.close_window()
                        self.streaming_windows.remove(window)

            messagebox.showinfo("Éxito", "Cámara eliminada.")

    def refresh_camera_list(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        for cid, name, url in self.camera_list_sorted:
            self.add_camera_to_list(self.scrollable_frame, cid, name, url)

    def prompt_add_camera(self):
        aw = Toplevel(self.root)
        aw.title("Agregar Cámara")
        aw.geometry("400x450")
        aw.resizable(False, False)
        aw.configure(bg="#F0F0F0")

        aw.grid_rowconfigure(0, weight=1)
        aw.grid_rowconfigure(1, weight=1)
        aw.grid_rowconfigure(2, weight=1)
        aw.grid_rowconfigure(3, weight=1)
        aw.grid_rowconfigure(4, weight=1)
        aw.grid_columnconfigure(0, weight=1)
        aw.grid_columnconfigure(1, weight=3)

        lbl_name = ttk.Label(aw, text="Nombre de la Cámara:", style="SubHeader.TLabel")
        lbl_name.grid(row=0, column=0, padx=10, pady=10, sticky="e")

        name_entry = ttk.Entry(aw, width=30)
        name_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        lbl_url = ttk.Label(aw, text="URL RTSP de la Cámara:", style="SubHeader.TLabel")
        lbl_url.grid(row=1, column=0, padx=10, pady=10, sticky="e")

        url_entry = ttk.Entry(aw, width=30)
        url_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        url_entry.insert(0, "")

        lbl_thumb = ttk.Label(aw, text="Miniatura:", style="SubHeader.TLabel")
        lbl_thumb.grid(row=2, column=0, padx=10, pady=10, sticky="e")

        thumb_label = ttk.Label(aw)
        thumb_label.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        thumb = self.get_camera_thumbnail("")
        if thumb is not None:
            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(thumb)
            img = img.resize((100, 75), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            thumb_label.config(image=imgtk)
            thumb_label.image = imgtk
        else:
            placeholder = Image.new('RGB', (100, 75), color='gray')
            imgtk = ImageTk.PhotoImage(image=placeholder)
            thumb_label.config(image=imgtk)
            thumb_label.image = imgtk

        def preview_thumb():
            url = url_entry.get().strip()
            thumb = self.get_camera_thumbnail(url)
            if thumb is not None:
                thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(thumb)
                img = img.resize((100, 75), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                thumb_label.config(image=imgtk)
                thumb_label.image = imgtk
            else:
                placeholder = Image.new('RGB', (100, 75), color='gray')
                imgtk = ImageTk.PhotoImage(image=placeholder)
                thumb_label.config(image=imgtk)
                thumb_label.image = imgtk

        btn_preview_thumb = ttk.Button(aw, text="Previsualizar Miniatura", command=preview_thumb)
        btn_preview_thumb.grid(row=3, column=1, padx=10, pady=5, sticky="w")

        def add_cam():
            name = name_entry.get().strip()
            url = url_entry.get().strip()
            if not name or not url:
                messagebox.showerror("Error", "Complete todos los campos.")
                return
            new_id = add_camera_db(self.cursor, self.conn, name, url)
            self.camera_list.append((new_id, name, url))
            self.camera_list_sorted = sorted(self.camera_list, key=lambda c: c[0])

            thumb = self.get_camera_thumbnail(url)
            if thumb is not None:
                thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(thumb)
                img = img.resize((100, 75), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.thumbnails[new_id] = imgtk
            else:
                img = Image.new('RGB', (100, 75), color='gray')
                imgtk = ImageTk.PhotoImage(image=img)
                self.thumbnails[new_id] = imgtk

            vs = VideoStream(url)
            vs.start()
            self.streams[new_id] = vs
            self.refresh_camera_list()
            messagebox.showinfo("Éxito", "Cámara agregada con éxito.")
            aw.destroy()

        btn_add_cam = ttk.Button(aw, text="Agregar Cámara", command=add_cam)
        btn_add_cam.grid(row=4, column=0, columnspan=2, padx=10, pady=20, sticky="ew")

    def import_cameras_json(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo JSON",
            filetypes=[("Archivos JSON", "*.json")]
        )
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            camaras = data.get("camaras", [])
            if not camaras:
                messagebox.showwarning("JSON vacío", "No se encontraron cámaras.")
                return
            for cam in camaras:
                name = cam.get("name")
                url = cam.get("url")
                if name and url:
                    new_id = add_camera_db(self.cursor, self.conn, name, url)
                    self.camera_list.append((new_id, name, url))
                    self.camera_list_sorted = sorted(self.camera_list, key=lambda c: c[0])
                    thumb = self.get_camera_thumbnail(url)
                    if thumb is not None:
                        thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(thumb)
                        img = img.resize((100, 75), Image.Resampling.LANCZOS)
                        imgtk = ImageTk.PhotoImage(image=img)
                        self.thumbnails[new_id] = imgtk
                    else:
                        img = Image.new('RGB', (100, 75), color='gray')
                        imgtk = ImageTk.PhotoImage(image=img)
                        self.thumbnails[new_id] = imgtk

                    vs = VideoStream(url)
                    vs.start()
                    self.streams[new_id] = vs
            self.refresh_camera_list()
            messagebox.showinfo("Importación", "¡Cámaras importadas con éxito!")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo importar: {e}")

    ############################################
    # Selección de cámaras para modo "Single"
    ############################################
    def select_cameras_single(self):
        if not self.camera_list:
            messagebox.showwarning("Sin Cámaras", "No hay cámaras para mostrar.")
            return
        selection_window = Toplevel(self.root)
        selection_window.title("Seleccionar Cámaras")
        selection_window.geometry("400x500")
        selection_window.resizable(False, False)
        selection_window.configure(bg="#F0F0F0")

        selection_window.grid_rowconfigure(0, weight=1)
        selection_window.grid_rowconfigure(1, weight=1)
        selection_window.grid_columnconfigure(0, weight=1)

        lbl = ttk.Label(selection_window,
                        text="Selecciona las cámaras que deseas ver individualmente:",
                        style="SubHeader.TLabel")
        lbl.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        camera_var = {}
        cameras_frame = ttk.Frame(selection_window, style="Main.TFrame")
        cameras_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        cameras_frame.grid_rowconfigure(0, weight=1)
        cameras_frame.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(cameras_frame, bg="#F0F0F0")
        scrollbar = ttk.Scrollbar(cameras_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style="Main.TFrame")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        cameras_frame.grid_rowconfigure(0, weight=1)
        cameras_frame.grid_columnconfigure(0, weight=1)

        for cid, name, url in self.camera_list:
            cam_frame = ttk.Frame(scrollable_frame, style="Main.TFrame")
            cam_frame.pack(anchor="w", padx=10, pady=5)

            if cid in self.thumbnails:
                imgtk = self.thumbnails[cid]
            else:
                img = Image.new('RGB', (50, 37), color='gray')
                imgtk = ImageTk.PhotoImage(image=img)
                self.thumbnails[cid] = imgtk

            lbl_thumb = ttk.Label(cam_frame, image=imgtk)
            lbl_thumb.image = imgtk
            lbl_thumb.pack(side="left", padx=(0, 10))

            var = tk.BooleanVar()
            chk = ttk.Checkbutton(cam_frame, text=name, variable=var)
            chk.pack(side="left")
            camera_var[cid] = var

        def select_all():
            for c_id in camera_var:
                camera_var[c_id].set(True)

        btn_select_all = ttk.Button(selection_window, text="Seleccionar Todas", command=select_all)
        btn_select_all.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        def open_single():
            selected_cids = [cid for cid, var in camera_var.items() if var.get()]
            if not selected_cids:
                messagebox.showerror("Error", "Seleccione al menos una cámara.")
                return
            selected_cameras = [c for c in self.camera_list if c[0] in selected_cids]
            monitor_name = self.selected_monitor.get()
            if monitor_name == "Todas las Pantallas":
                monitors = self.get_monitors_list()
                if "Todas las Pantallas" in monitors:
                    monitors = monitors[1:]  # Excluir la opción "Todas las Pantallas"
                for monitor in monitors:
                    monitor_geom = self.get_monitor_geometry(monitor)
                    window = FullscreenWindow(
                        parent=self,
                        mode="Single",
                        cameras=selected_cameras,
                        streams=self.streams,
                        frame_rate=self.frame_rate,
                        auto_rotate=self.auto_rotate,
                        rotation_time=self.rotation_time,
                        monitor_geom=monitor_geom
                    )
                    self.streaming_windows.append(window)
            else:
                monitor_geom = self.get_monitor_geometry(monitor_name)
                window = FullscreenWindow(
                    parent=self,
                    mode="Single",
                    cameras=selected_cameras,
                    streams=self.streams,
                    frame_rate=self.frame_rate,
                    auto_rotate=self.auto_rotate,
                    rotation_time=self.rotation_time,
                    monitor_geom=monitor_geom
                )
                self.streaming_windows.append(window)
            selection_window.destroy()

        btn_ver = ttk.Button(selection_window, text="Ver Cámaras", command=open_single)
        btn_ver.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

    ############################################
    # Selección de cámaras para modo "Grid4"
    ############################################
    def select_cameras_grid(self, grid_size):
        if not self.camera_list:
            messagebox.showwarning("Sin Cámaras", "No hay cámaras para mostrar.")
            return
        selection_window = Toplevel(self.root)
        selection_window.title(f"Seleccionar Cámaras para Cuadrícula {grid_size}")
        selection_window.geometry("500x600")
        selection_window.resizable(False, False)
        selection_window.configure(bg="#F0F0F0")

        selection_window.grid_rowconfigure(0, weight=1)
        selection_window.grid_rowconfigure(1, weight=1)
        selection_window.grid_columnconfigure(0, weight=1)

        lbl = ttk.Label(selection_window,
                        text=f"Selecciona hasta 20 cámaras para Cuadrícula {grid_size}:",
                        style="SubHeader.TLabel")
        lbl.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        cameras_frame = ttk.Frame(selection_window, style="Main.TFrame")
        cameras_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        cameras_frame.grid_rowconfigure(0, weight=1)
        cameras_frame.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(cameras_frame, bg="#F0F0F0")
        scrollbar = ttk.Scrollbar(cameras_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style="Main.TFrame")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        cameras_frame.grid_rowconfigure(0, weight=1)
        cameras_frame.grid_columnconfigure(0, weight=1)

        self.grid_selected_cameras = 0
        max_selection = 20
        camera_vars = {}

        def on_check(cid, var):
            if var.get():
                self.grid_selected_cameras += 1
                if self.grid_selected_cameras > max_selection:
                    var.set(False)
                    self.grid_selected_cameras -= 1
                    messagebox.showerror(
                        "Error",
                        f"Puedes seleccionar hasta {max_selection} cámaras para la cuadrícula."
                    )
            else:
                self.grid_selected_cameras -= 1

        for cid, name, url in self.camera_list:
            cam_frame = ttk.Frame(scrollable_frame, style="Main.TFrame")
            cam_frame.pack(anchor="w", padx=10, pady=5)

            if cid in self.thumbnails:
                imgtk = self.thumbnails[cid]
            else:
                img = Image.new('RGB', (50, 37), color='gray')
                imgtk = ImageTk.PhotoImage(image=img)
                self.thumbnails[cid] = imgtk

            lbl_thumb = ttk.Label(cam_frame, image=imgtk)
            lbl_thumb.image = imgtk
            lbl_thumb.pack(side="left", padx=(0, 10))

            var = tk.BooleanVar()
            chk = ttk.Checkbutton(
                cam_frame, text=name, variable=var,
                command=lambda c=cid, v=var: on_check(c, v)
            )
            chk.pack(side="left")

            camera_vars[cid] = var

        def select_all_grid():
            sorted_cids = [c[0] for c in self.camera_list]
            selected_count = 0
            for c_id in sorted_cids:
                if selected_count < 20:
                    camera_vars[c_id].set(True)
                    selected_count += 1
                else:
                    camera_vars[c_id].set(False)
            self.grid_selected_cameras = selected_count

        btn_select_all = ttk.Button(selection_window, text="Seleccionar Todas (máx 20)", command=select_all_grid)
        btn_select_all.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        def open_grid():
            selected_cids = [cid for cid, var in camera_vars.items() if var.get()]
            if len(selected_cids) == 0:
                messagebox.showerror("Error", "Selecciona al menos una cámara.")
                return
            if len(selected_cids) > max_selection:
                messagebox.showerror("Error", f"Puedes seleccionar hasta {max_selection} cámaras.")
                return

            selected_cameras = [c for c in self.camera_list if c[0] in selected_cids]
            monitor_name = self.selected_monitor.get()
            if monitor_name == "Todas las Pantallas":
                monitors = self.get_monitors_list()
                if "Todas las Pantallas" in monitors:
                    monitors = monitors[1:]  # Excluir la opción "Todas las Pantallas"
                for monitor in monitors:
                    monitor_geom = self.get_monitor_geometry(monitor)
                    window = FullscreenWindow(
                        parent=self,
                        mode=f"Grid{grid_size}",
                        cameras=selected_cameras,
                        streams=self.streams,
                        frame_rate=self.frame_rate,
                        auto_rotate=self.auto_rotate,
                        rotation_time=self.rotation_time,
                        monitor_geom=monitor_geom
                    )
                    self.streaming_windows.append(window)
            else:
                monitor_geom = self.get_monitor_geometry(monitor_name)
                window = FullscreenWindow(
                    parent=self,
                    mode=f"Grid{grid_size}",
                    cameras=selected_cameras,
                    streams=self.streams,
                    frame_rate=self.frame_rate,
                    auto_rotate=self.auto_rotate,
                    rotation_time=self.rotation_time,
                    monitor_geom=monitor_geom
                )
                self.streaming_windows.append(window)
            selection_window.destroy()

        btn_ver = ttk.Button(selection_window, text="Ver Cuadrícula", command=open_grid)
        btn_ver.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        self.camera_vars = camera_vars

    def apply_frame_rate(self):
        selected_fps = self.frame_rate_var.get()
        if selected_fps not in [1, 3, 5, 10, 15, 20, 30, 60]:
            messagebox.showerror("Error", "Selecciona una velocidad de Frame válida.")
            return
        self.frame_rate = selected_fps
        for window in self.streaming_windows:
            window.update_frame_rate(self.frame_rate)
        messagebox.showinfo("Éxito", f"Velocidad de Frame actualizada a {self.frame_rate} FPS.")

    def apply_rotation_time(self):
        try:
            val = int(self.rotation_entry.get())
            if val <= 0:
                raise ValueError
            self.rotation_time = val
            for window in self.streaming_windows:
                window.rotation_time = self.rotation_time
                if window.auto_rotate:
                    if window.rotation_job:
                        window.root.after_cancel(window.rotation_job)
                    window.start_auto_rotation()
            messagebox.showinfo("Éxito", f"Tiempo de rotación actualizado a {self.rotation_time} segundos.")
        except ValueError:
            messagebox.showerror("Error", "Ingresa un número entero positivo.")
            self.rotation_entry.delete(0, tk.END)
            self.rotation_entry.insert(0, str(self.rotation_time))

    def get_monitors_list(self):
        """
        Retorna una lista de nombres para las pantallas detectadas.
        Si no se detecta screeninfo, retorna una única opción "Principal".
        """
        monitors_list = ["Todas las Pantallas"]
        if get_monitors:
            for idx, m in enumerate(get_monitors()):
                monitors_list.append(f"Pantalla {idx+1}")
        else:
            monitors_list.append("Principal")
        return monitors_list

    def get_monitor_geometry(self, monitor_name):
        """
        Dado el nombre de la pantalla (ej: "Pantalla 1"), retorna un diccionario con
        la geometría: {"x": pos_x, "y": pos_y, "width": w, "height": h}.
        Si no se pudo detectar, retorna None.
        """
        if get_monitors:
            monitors = list(get_monitors())
            try:
                if monitor_name == "Todas las Pantallas":
                    return None  # No aplicable para una sola ventana
                idx = int(monitor_name.split()[1]) - 1
                m = monitors[idx]
                geom = {"x": m.x, "y": m.y, "width": m.width, "height": m.height}
                print(f"Monitor '{monitor_name}': {geom}")  # DEBUG: Verificar geometría
                return geom
            except Exception as e:
                print(f"Error obteniendo la geometría de la pantalla: {e}")
        return None

    ################################################
    # Función para Reiniciar Todas las Cámaras
    ################################################
    def restart_all_cameras(self):
        """
        Reinicia todos los streams de las cámaras.
        """
        confirm = messagebox.askyesno("Confirmar", "¿Estás seguro de que deseas reiniciar todas las cámaras?")
        if not confirm:
            return

        failed_restarts = []
        for cid, vs in self.streams.items():
            try:
                # Detener el stream actual
                vs.stop()

                # Obtener la URL de la cámara desde la lista
                cam = next((c for c in self.camera_list if c[0] == cid), None)
                if cam:
                    _, _, url = cam
                    # Crear y comenzar un nuevo stream
                    new_vs = VideoStream(url)
                    new_vs.start()
                    self.streams[cid] = new_vs
                else:
                    failed_restarts.append(cid)
            except Exception as e:
                print(f"Error al reiniciar la cámara ID {cid}: {e}")
                failed_restarts.append(cid)
        
        if failed_restarts:
            messagebox.showerror("Errores al Reiniciar", f"No se pudieron reiniciar las cámaras con IDs: {failed_restarts}")
        else:
            messagebox.showinfo("Éxito", "Todas las cámaras han sido reiniciadas correctamente.")

    def __del__(self):
        for s in self.streams.values():
            s.stop()
        self.conn.close()

############################################
# INICIAR LA APLICACIÓN
############################################
def main():
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
