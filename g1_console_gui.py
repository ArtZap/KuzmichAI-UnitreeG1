#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
g1_console — нативный GUI консоли робота G1 (PySide6).

Тонкий клиент: сам ничего не умеет, всё делает g1_daemon по TCP 15100 (NDJSON).
Именно поэтому многопользовательность и уровни допуска вообще работают —
права проверяет демон, а не кнопки в интерфейсе.

Вкладки:
  BIOS            — температуры/токи/напряжения моторов, IMU, батарея, Jetson, алерты
  Модули          — список доступных модулей, запуск/остановка, логи
  Моё пространство— добавить свой модуль (манифест + код) в личный каталог
  Админ           — сессии и журнал (только уровень «админ»)

Установка:  pip install PySide6
Запуск:     ./g1_console_gui.py --host 192.168.1.102
"""
import argparse, json, os, socket, sys, threading, time

try:
    from PySide6.QtCore import Qt, QObject, Signal, QTimer, QRectF
    from PySide6.QtGui import QColor, QFont, QBrush, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QListWidget,
        QListWidgetItem, QDialog, QLineEdit, QFormLayout, QComboBox, QCheckBox,
        QPlainTextEdit, QMessageBox, QHeaderView, QProgressBar, QGroupBox, QSplitter,
        QDialogButtonBox, QSpinBox, QDoubleSpinBox, QStatusBar, QScrollArea, QSlider,
    )
except ImportError:
    sys.exit('Нужен PySide6:  pip install PySide6')

COL_OK, COL_WARN, COL_CRIT, COL_DIM = QColor('#2e7d32'), QColor('#ef6c00'), QColor('#c62828'), QColor('#9e9e9e')
# Кисти создаём один раз: в горячем цикле обновления таблицы QBrush(color) на
# каждую ячейку каждый кадр — это тысячи объектов в секунду на пустом месте.
BR_OK, BR_WARN, BR_CRIT, BR_DIM = QBrush(COL_OK), QBrush(COL_WARN), QBrush(COL_CRIT), QBrush(COL_DIM)


# ─────────────────────────── связь с демоном ───────────────────────────

# Версия клиента. Растёт, когда в GUI появляется то, чего старый клиент не
# покажет. Демон сравнивает её со своей и пишет предупреждение: иначе оператор
# со старым файлом видит меньше вкладок и молча считает, что так и надо, —
# ровно это и случилось с панелями «Сюжет», «Карта» и «Руки».
CLIENT_VERSION = 10


class Link(QObject):
    """Сокет в отдельном потоке; сообщения приезжают в GUI через сигналы Qt."""
    message = Signal(dict)
    status = Signal(str, bool)   # текст, connected

    def __init__(self, host, port, user, token):
        super().__init__()
        self.host, self.port, self.user, self.token = host, port, user, token
        self.sock = None
        self.alive = True
        self.last_hint = ''
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        backoff = 1.0
        while self.alive:
            try:
                self.status.emit('подключение к %s:%d…' % (self.host, self.port), False)
                s = socket.create_connection((self.host, self.port), timeout=10)
                s.settimeout(None)
                with self.lock:
                    self.sock = s
                self._send_raw({'op': 'hello', 'user': self.user, 'token': self.token,
                                'client': CLIENT_VERSION})
                backoff = 1.0
                self.last_hint = ''
                buf = b''
                while self.alive:
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                    while b'\n' in buf:
                        line, buf = buf.split(b'\n', 1)
                        if line.strip():
                            try:
                                self.message.emit(json.loads(line.decode('utf-8')))
                            except Exception as e:
                                print('битая строка от демона: %r' % e)
            except Exception as e:
                # Отказ в соединении на localhost почти всегда значит одно: демон
                # не запущен. Писать про «нет связи» тут бесполезно — человек
                # смотрит на пустое окно и не понимает, что делать.
                if isinstance(e, ConnectionRefusedError) and self.host in ('127.0.0.1', 'localhost'):
                    self.last_hint = 'демон не запущен — закройте окно и выполните ./show_console.sh'
                else:
                    self.last_hint = 'нет связи с %s:%d — %s' % (self.host, self.port, e)
                self.status.emit(self.last_hint, False)
            with self.lock:
                self.sock = None
            if not self.alive:
                break
            self.status.emit(self.last_hint or ('переподключение через %.0f с…' % backoff), False)
            time.sleep(backoff)
            backoff = min(backoff * 2, 10.0)

    def _send_raw(self, obj):
        with self.lock:
            if not self.sock:
                return False
            try:
                self.sock.sendall((json.dumps(obj, ensure_ascii=False) + '\n').encode())
                return True
            except Exception:
                return False

    def send(self, obj):
        if not self._send_raw(obj):
            self.status.emit('команда не ушла — нет связи', False)

    def close(self):
        self.alive = False
        with self.lock:
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass


# ─────────────────────────────── вход ───────────────────────────────

class LoginDialog(QDialog):
    def __init__(self, host, port, user):
        super().__init__()
        self.setWindowTitle('Вход в консоль G1')
        f = QFormLayout(self)
        self.host = QLineEdit(host)
        self.port = QLineEdit(str(port))
        self.user = QLineEdit(user)
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.Password)
        f.addRow('Робот (хост):', self.host)
        f.addRow('Порт:', self.port)
        f.addRow('Пользователь:', self.user)
        f.addRow('Токен:', self.token)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)
        self.token.setFocus()

    def values(self):
        return self.host.text().strip(), int(self.port.text() or 15100), self.user.text().strip(), self.token.text()


# ──────────────────────────── вкладка BIOS ────────────────────────────

class NumItem(QTableWidgetItem):
    """Ячейка, которая сортируется по числу, а не по строке.
    Без этого «9.5» оказывается больше «10.2», и сортировка по температуре врёт."""

    def __lt__(self, other):
        a, b = self.data(Qt.UserRole), other.data(Qt.UserRole)
        if a is None or b is None:
            return super().__lt__(other)
        return a < b


class BiosTab(QWidget):
    # T°, макс — по ней срабатывают пороги; T°, датчики — оба сырых датчика мотора
    COLS = ['#', 'Сустав', 'Группа', 'T°, макс', 'T°, датчики', 'q, рад', 'dq, рад/с', 'τ, Н·м', 'U, В', 'Ошибка']

    def __init__(self):
        super().__init__()
        self.thresholds = {}
        self.group_of = {}
        self.rows = {}       # индекс мотора -> список QTableWidgetItem этой строки
        self.sev_cache = {}  # индекс мотора -> прошлая тяжесть, чтобы не трогать шрифт зря
        self.bar_cache = {}  # имя полоски Jetson -> прошлый цвет, чтобы не парсить стиль зря
        self.alert_sig = None
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.lbl_summary = QLabel('нет данных')
        self.lbl_summary.setFont(QFont('', 13, QFont.Bold))
        top.addWidget(self.lbl_summary)
        top.addStretch()
        self.lbl_age = QLabel('')
        top.addWidget(self.lbl_age)
        root.addLayout(top)

        split = QSplitter(Qt.Horizontal)

        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        # ВАЖНО для скорости: ResizeToContents заставляет Qt перемерять КАЖДУЮ
        # ячейку при любом изменении данных. На живой таблице 29×10 это главный
        # источник тормозов. Меряем один раз после первого заполнения строк
        # (см. update_data), дальше держим фиксированную ширину.
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        split.addWidget(self.table)

        right = QWidget()
        rl = QVBoxLayout(right)

        self.box_alerts = QGroupBox('Алерты')
        al = QVBoxLayout(self.box_alerts)
        self.alerts = QListWidget()
        # Выделение здесь не нужно (с алертом всё равно нечего делать), а мешает:
        # выделенная строка рисуется цветом выделения и теряет красную/жёлтую
        # подсветку тяжести — критичный алерт визуально становится обычным.
        self.alerts.setSelectionMode(QListWidget.NoSelection)
        self.alerts.setFocusPolicy(Qt.NoFocus)
        al.addWidget(self.alerts)
        rl.addWidget(self.box_alerts, 2)

        self.box_imu = QGroupBox('IMU / Батарея')
        g = QGridLayout(self.box_imu)
        self.imu_labels = {}
        for r, key in enumerate(['Крен/тангаж/рыскание', 'Температура IMU',
                                 'Заряд', 'Напряжение', 'Ток',
                                 'Износ (SOH)', 'Температура батареи']):
            g.addWidget(QLabel(key + ':'), r, 0)
            lab = QLabel('—')
            self.imu_labels[key] = lab
            g.addWidget(lab, r, 1)
        rl.addWidget(self.box_imu)

        self.box_host = QGroupBox('Jetson')
        hg = QGridLayout(self.box_host)
        self.host_widgets = {}
        rows = [('CPU, °C', 'cpu_temp', 100), ('GPU, °C', 'gpu_temp', 100),
                ('CPU, %', 'cpu_pct', 100), ('GPU, %', 'gpu_pct', 100),
                ('Память, %', 'mem_pct', 100), ('Диск, %', 'disk_pct', 100)]
        for r, (label, key, mx) in enumerate(rows):
            hg.addWidget(QLabel(label), r, 0)
            bar = QProgressBar()
            bar.setRange(0, mx)
            bar.setFormat('%v')
            bar.setAlignment(Qt.AlignCenter)
            self.host_widgets[key] = bar
            hg.addWidget(bar, r, 1)
        self.lbl_power = QLabel('—')
        hg.addWidget(QLabel('Питание:'), len(rows), 0)
        hg.addWidget(self.lbl_power, len(rows), 1)
        rl.addWidget(self.box_host)
        rl.addStretch()

        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setSizes([1000, 420])
        root.addWidget(split, 1)   # без stretch таблица не забирает высоту и вверху зияет пустота

    def configure(self, cfg):
        self.thresholds = cfg.get('thresholds', {})
        self.group_of = {}
        for gname, idxs in cfg.get('joint_groups', {}).items():
            for i in idxs:
                self.group_of[i] = gname

    def motor_th(self, i):
        t = self.thresholds.get('motor_temp', {})
        return t.get(self.group_of.get(i)) or t.get('default', {'warn': 60, 'crit': 75})

    def update_data(self, snap, alerts):
        names = snap.get('joint_names', [])
        motors = snap.get('motors', [])
        age = snap.get('lowstate_age')
        if age is None:
            self.lbl_age.setText('телеметрия робота: нет данных')
            self.lbl_age.setStyleSheet('color:#c62828')
        else:
            self.lbl_age.setText('телеметрия робота: %.1f с назад · tick %s' % (age, snap.get('tick')))
            self.lbl_age.setStyleSheet('color:#c62828' if age > 3 else 'color:#2e7d32')

        # Строки адресуем по СОХРАНЁННЫМ ссылкам на ячейки, а не по номеру строки:
        # после сортировки пользователем строка мотора уезжает, и запись по индексу
        # перемешала бы данные между суставами. Ссылки на QTableWidgetItem переезжают
        # вместе со строкой и остаются валидными.
        if len(self.rows) != len(motors):
            self.table.setSortingEnabled(False)
            self.table.setRowCount(0)
            self.rows = {}
            self.table.setRowCount(len(motors))
            for r, m in enumerate(motors):
                i = m.get('i', r)
                cells = []
                for c in range(len(self.COLS)):
                    it = NumItem() if c != 1 and c != 2 else QTableWidgetItem()
                    if c not in (1, 2):
                        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.table.setItem(r, c, it)
                    cells.append(it)
                cells[0].setText(str(i)); cells[0].setData(Qt.UserRole, i)
                cells[1].setText(names[i] if i < len(names) else 'motor%d' % i)
                cells[2].setText(self.group_of.get(i, ''))
                self.rows[i] = cells
            self.table.setSortingEnabled(True)
            # по умолчанию — штатный порядок суставов, а не то, что Qt выберет сам
            self.table.sortItems(0, Qt.AscendingOrder)
            # ширины подбираем ОДИН раз по заполненным строкам и фиксируем
            hdr = self.table.horizontalHeader()
            hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
            self.table.resizeColumnsToContents()
            hdr.setSectionResizeMode(QHeaderView.Interactive)
            hdr.setSectionResizeMode(1, QHeaderView.Stretch)
            self.sev_cache = {}

        # Сортировку на время записи выключаем: с включённой Qt пересортировывает
        # таблицу на КАЖДЫЙ setData, то есть ~200 сортировок на кадр вместо одной.
        # Порядок и выбранная пользователем колонка при этом сохраняются.
        self.table.setSortingEnabled(False)
        self.table.setUpdatesEnabled(False)

        hottest, hot_t = None, -1e9
        for m in motors:
            i = m.get('i')
            cells = self.rows.get(i)
            if not cells:
                continue
            temps = m.get('temp') or [None, None]
            tmax = max([t for t in temps if t is not None], default=None)
            th = self.motor_th(i)
            if tmax is not None and tmax > hot_t:
                hot_t, hottest = tmax, (names[i] if i < len(names) else str(i))
            nums = [(3, tmax), (5, m.get('q')), (6, m.get('dq')), (7, m.get('tau')), (8, m.get('vol'))]
            fmts = {3: '%.1f', 5: '%+.3f', 6: '%+.3f', 7: '%+.2f', 8: '%.2f'}
            for c, v in nums:
                cells[c].setText(fmts[c] % v if v is not None else '—')
                cells[c].setData(Qt.UserRole, v)
            raw = [t for t in temps if t is not None]
            cells[4].setText(' / '.join('%.1f' % t for t in raw) if raw else '—')
            cells[4].setData(Qt.UserRole, min(raw) if raw else None)
            cells[9].setText('—' if not m.get('err') else str(m['err']))
            cells[9].setData(Qt.UserRole, m.get('err') or 0)
            # цвет и жирность трогаем только при смене состояния: setFont каждый
            # кадр заставляет Qt пересчитывать метрики строки
            sev = None
            if tmax is not None:
                sev = 'crit' if tmax >= th['crit'] else ('warn' if tmax >= th['warn'] else 'ok')
            err = bool(m.get('err'))
            if self.sev_cache.get(i) != (sev, err):
                if sev:
                    cells[3].setForeground(BR_CRIT if sev == 'crit' else (BR_WARN if sev == 'warn' else BR_OK))
                    f = cells[3].font()
                    f.setBold(sev != 'ok')
                    cells[3].setFont(f)
                cells[9].setForeground(BR_CRIT if err else BR_DIM)
                self.sev_cache[i] = (sev, err)

        self.table.setUpdatesEnabled(True)
        self.table.setSortingEnabled(True)

        crit = sum(1 for a in alerts if a['sev'] == 'crit')
        warn = sum(1 for a in alerts if a['sev'] == 'warn')
        if crit:
            self.lbl_summary.setText('КРИТИЧНО: %d  ·  предупреждений: %d' % (crit, warn))
            self.lbl_summary.setStyleSheet('color:#c62828')
        elif warn:
            self.lbl_summary.setText('Предупреждений: %d' % warn)
            self.lbl_summary.setStyleSheet('color:#ef6c00')
        else:
            self.lbl_summary.setText('Все узлы в норме · самый горячий: %s %.1f°C'
                                     % (hottest, hot_t) if hottest else 'Все узлы в норме')
            self.lbl_summary.setStyleSheet('color:#2e7d32')

        # Список алертов меняется редко — перестраиваем только когда он реально
        # изменился, а не пять раз в секунду вхолостую.
        sig = tuple((a['id'], a['sev'], a['text']) for a in alerts)
        if sig != self.alert_sig:
            self.alert_sig = sig
            self.alerts.clear()
            for a in alerts:
                it = QListWidgetItem('%s  %s  (%s)' % ('⛔' if a['sev'] == 'crit' else '⚠️', a['text'],
                                                       time.strftime('%H:%M:%S', time.localtime(a['since']))))
                it.setForeground(BR_CRIT if a['sev'] == 'crit' else BR_WARN)
                self.alerts.addItem(it)
            if not alerts:
                self.alerts.addItem(QListWidgetItem('— пусто —'))

        imu = snap.get('imu') or {}
        rpy = imu.get('rpy') or []
        self.imu_labels['Крен/тангаж/рыскание'].setText(
            '  '.join('%+.3f' % v for v in rpy) if rpy else '—')
        self.imu_labels['Температура IMU'].setText('%.1f °C' % imu['temp'] if imu.get('temp') is not None else '—')
        bms = snap.get('bms') or {}
        self.imu_labels['Заряд'].setText('%.0f %%' % bms['soc'] if bms.get('soc') is not None else 'н/д')
        self.imu_labels['Напряжение'].setText('%.2f В' % bms['voltage'] if bms.get('voltage') is not None else 'н/д')
        self.imu_labels['Ток'].setText('%.2f А' % bms['current'] if bms.get('current') is not None else 'н/д')
        self.imu_labels['Износ (SOH)'].setText(
            '%s %% · %s циклов' % (bms['soh'], bms.get('cycle', '?')) if bms.get('soh') is not None else 'н/д')
        self.imu_labels['Температура батареи'].setText(
            '%s °C' % bms['temp'] if bms.get('temp') is not None else 'н/д')

        h = snap.get('host') or {}
        mem_pct = None
        if h.get('mem_used_mb') and h.get('mem_total_mb'):
            mem_pct = 100.0 * h['mem_used_mb'] / h['mem_total_mb']
        for key, val in (('cpu_temp', h.get('cpu_temp')), ('gpu_temp', h.get('gpu_temp')),
                         ('cpu_pct', h.get('cpu_pct')), ('gpu_pct', h.get('gpu_pct')),
                         ('mem_pct', mem_pct), ('disk_pct', h.get('disk_pct'))):
            bar = self.host_widgets[key]
            if val is None:
                bar.setValue(0)
                bar.setFormat('н/д')
            else:
                bar.setValue(int(val))
                bar.setFormat('%.0f' % val)
                warn_at = 70 if 'temp' in key else 85
                crit_at = 85 if 'temp' in key else 95
                col = '#c62828' if val >= crit_at else ('#ef6c00' if val >= warn_at else '#2e7d32')
                # setStyleSheet заново разбирает CSS и перестраивает стиль виджета —
                # дорого. Трогаем только когда цвет действительно сменился.
                if self.bar_cache.get(key) != col:
                    self.bar_cache[key] = col
                    bar.setStyleSheet('QProgressBar::chunk{background:%s}' % col)
        pw = h.get('power_mw')
        self.lbl_power.setText(('%.1f Вт' % (pw / 1000.0)) if pw else 'н/д')


# ─────────────────────────── вкладка «Модули» ───────────────────────────

class ParamsDialog(QDialog):
    def __init__(self, mod):
        super().__init__()
        self.setWindowTitle('Параметры: %s' % mod.get('title', mod['name']))
        self.widgets = {}
        f = QFormLayout(self)
        for p in mod.get('params', []):
            t = p.get('type', 'str')
            if t == 'int':
                w = QSpinBox(); w.setRange(int(p.get('min', -10 ** 6)), int(p.get('max', 10 ** 6)))
                w.setValue(int(p.get('default', 0)))
            elif t == 'float':
                w = QDoubleSpinBox(); w.setDecimals(3)
                w.setRange(float(p.get('min', -1e6)), float(p.get('max', 1e6)))
                w.setSingleStep(0.05); w.setValue(float(p.get('default', 0)))
            elif t == 'bool':
                w = QCheckBox(); w.setChecked(bool(p.get('default', False)))
            elif t == 'choice':
                w = QComboBox(); w.addItems([str(c) for c in p.get('choices', [])])
                if p.get('default') is not None:
                    w.setCurrentText(str(p['default']))
            else:
                w = QLineEdit(str(p.get('default', '')))
            self.widgets[p['name']] = (w, t)
            f.addRow(p.get('title', p['name']) + ':', w)
        if not mod.get('params'):
            f.addRow(QLabel('у модуля нет параметров'))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        f.addRow(bb)

    def values(self):
        out = {}
        for name, (w, t) in self.widgets.items():
            if t == 'int':
                out[name] = w.value()
            elif t == 'float':
                out[name] = w.value()
            elif t == 'bool':
                out[name] = 'true' if w.isChecked() else 'false'
            elif t == 'choice':
                out[name] = w.currentText()
            else:
                out[name] = w.text()
        return out


class ModulesTab(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.modules = []
        self.running = {}
        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        self.btn_start = QPushButton('Запустить')
        self.btn_stop = QPushButton('Остановить')
        self.btn_logs = QPushButton('Обновить лог')
        self.btn_delete = QPushButton('Удалить свой модуль')
        for b in (self.btn_start, self.btn_stop, self.btn_logs, self.btn_delete):
            bar.addWidget(b)
        bar.addStretch()
        self.btn_motion = QPushButton('Взять управление движением')
        self.btn_motion.setCheckable(True)
        bar.addWidget(self.btn_motion)
        root.addLayout(bar)

        split = QSplitter(Qt.Vertical)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(['Модуль', 'Название', 'Тип', 'Владелец', 'Допуск', 'Движение', 'Состояние'])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        split.addWidget(self.table)

        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setFont(QFont('Menlo' if sys.platform == 'darwin' else 'Monospace', 11))
        self.logs.setPlaceholderText('Лог выбранного модуля')
        split.addWidget(self.logs)
        split.setSizes([420, 320])
        root.addWidget(split)

        self.btn_start.clicked.connect(self.do_start)
        self.btn_stop.clicked.connect(self.do_stop)
        self.btn_logs.clicked.connect(self.do_logs)
        self.btn_delete.clicked.connect(self.do_delete)
        self.btn_motion.clicked.connect(self.do_motion)
        self.table.itemSelectionChanged.connect(self.do_logs)

    def selected(self):
        r = self.table.currentRow()
        if r < 0 or r >= len(self.modules):
            return None
        return self.modules[r]

    def key_of(self, mod):
        return mod['name'] if mod['scope'] == 'system' else '%s/%s' % (mod['owner'], mod['name'])

    def set_modules(self, mods, running):
        self.modules = mods
        self.running = {r['key']: r for r in running}
        cur = self.table.currentRow()
        self.table.setRowCount(len(mods))
        for r, m in enumerate(mods):
            key = self.key_of(m)
            rs = self.running.get(key)
            if rs and rs['alive']:
                state = 'работает (pid %d, %.0f с)' % (rs['pid'], rs['uptime'])
            elif rs:
                state = 'остановлен (код %s)' % rs['exit_code']
            else:
                state = '—'
            vals = [m['name'], m['title'], 'системный' if m['scope'] == 'system' else 'личный',
                    m.get('owner') or '—', m['min_level_name'],
                    'да' if m['requires_motion'] else '', state]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c == 6 and rs and rs['alive']:
                    it.setForeground(QBrush(COL_OK))
                self.table.setItem(r, c, it)
        if 0 <= cur < len(mods):
            self.table.selectRow(cur)

    def do_start(self):
        m = self.selected()
        if not m:
            return
        d = ParamsDialog(m)
        if m.get('params') and d.exec() != QDialog.Accepted:
            return
        params = d.values() if m.get('params') else {}
        self.win.link.send({'op': 'module.start', 'key': self.key_of(m), 'params': params})

    def do_stop(self):
        m = self.selected()
        if m:
            self.win.link.send({'op': 'module.stop', 'key': self.key_of(m)})

    def do_logs(self):
        m = self.selected()
        if m and self.key_of(m) in self.running:
            self.win.link.send({'op': 'module.logs', 'key': self.key_of(m), 'n': 300})

    def do_delete(self):
        m = self.selected()
        if not m:
            return
        if m['scope'] != 'user':
            QMessageBox.warning(self, 'Нельзя', 'Удалять можно только свои модули.')
            return
        if QMessageBox.question(self, 'Удалить', 'Удалить модуль «%s»?' % m['name']) == QMessageBox.Yes:
            self.win.link.send({'op': 'module.delete', 'key': self.key_of(m)})

    def do_motion(self):
        self.win.link.send({'op': 'motion.acquire' if self.btn_motion.isChecked() else 'motion.release'})

    def show_logs(self, key, lines):
        m = self.selected()
        if not m or self.key_of(m) != key:
            return
        self.logs.setPlainText('\n'.join('%s  %s' % (time.strftime('%H:%M:%S', time.localtime(l['t'])), l['line'])
                                         for l in lines))
        self.logs.verticalScrollBar().setValue(self.logs.verticalScrollBar().maximum())


# ────────────────────── вкладка «Моё пространство» ──────────────────────

TEMPLATE = '''#!/usr/bin/env python3
# Модуль пользователя. Запускается демоном как обычный процесс.
# Переменные окружения: G1_USER, G1_USER_DIR (личный каталог), G1_MODULE.
# Всё, что печатается в stdout, попадает в лог модуля в GUI.
import os, time

print('привет из модуля, пользователь %s' % os.environ.get('G1_USER'))
print('личный каталог: %s' % os.environ.get('G1_USER_DIR'))
n = 0
while True:
    n += 1
    print('тик %d' % n, flush=True)
    time.sleep(1)
'''


class MySpaceTab(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        root = QVBoxLayout(self)
        self.lbl_dir = QLabel('личный каталог: —')
        root.addWidget(self.lbl_dir)

        form = QGroupBox('Новый модуль')
        f = QFormLayout(form)
        self.name = QLineEdit('my_module')
        self.title = QLineEdit('Мой модуль')
        self.descr = QLineEdit('')
        self.entry = QLineEdit('main.py')
        self.level = QComboBox()
        self.level.addItems(['новичок', 'младший', 'специалист', 'старший специалист', 'админ'])
        self.level.setCurrentText('младший')
        self.motion = QCheckBox('модуль двигает робота (нужен токен движения)')
        f.addRow('Имя (латиницей):', self.name)
        f.addRow('Название:', self.title)
        f.addRow('Описание:', self.descr)
        f.addRow('Файл запуска:', self.entry)
        f.addRow('Минимальный допуск:', self.level)
        f.addRow('', self.motion)
        root.addWidget(form)

        root.addWidget(QLabel('Код модуля:'))
        self.code = QPlainTextEdit(TEMPLATE)
        self.code.setFont(QFont('Menlo' if sys.platform == 'darwin' else 'Monospace', 11))
        root.addWidget(self.code, 1)

        bar = QHBoxLayout()
        self.btn_add = QPushButton('Добавить в моё пространство')
        bar.addWidget(self.btn_add)
        bar.addStretch()
        root.addLayout(bar)
        self.btn_add.clicked.connect(self.do_add)

    def do_add(self):
        name = self.name.text().strip()
        entry = self.entry.text().strip() or 'main.py'
        if not name:
            return
        man = {'name': name, 'title': self.title.text().strip() or name,
               'description': self.descr.text().strip(), 'version': '0.1',
               'entrypoint': ['python3', entry],
               'min_level': ['новичок', 'младший', 'специалист', 'старший специалист', 'админ']
                            .index(self.level.currentText()) + 1,
               'requires_motion': self.motion.isChecked(), 'params': []}
        self.win.link.send({'op': 'module.register', 'manifest': man, 'source': self.code.toPlainText()})


# ─────────────────────────── вкладка «Админ» ───────────────────────────

class AdminTab(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        b1 = QPushButton('Обновить сессии')
        b2 = QPushButton('Обновить журнал')
        bar.addWidget(b1); bar.addWidget(b2); bar.addStretch()
        root.addLayout(bar)
        self.sessions = QTableWidget(0, 3)
        self.sessions.setHorizontalHeaderLabels(['Пользователь', 'Уровень', 'Адрес'])
        self.sessions.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        root.addWidget(self.sessions)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setFont(QFont('Menlo' if sys.platform == 'darwin' else 'Monospace', 11))
        root.addWidget(self.log, 1)
        b1.clicked.connect(lambda: self.win.link.send({'op': 'admin.sessions'}))
        b2.clicked.connect(lambda: self.win.link.send({'op': 'admin.log'}))

    def set_sessions(self, data):
        self.sessions.setRowCount(len(data))
        for r, s in enumerate(data):
            for c, v in enumerate([s['user'], str(s['level']), s['peer']]):
                self.sessions.setItem(r, c, QTableWidgetItem(v))

    def set_log(self, data):
        self.log.setPlainText('\n'.join(
            '%s  %s%s' % (time.strftime('%H:%M:%S', time.localtime(e['t'])),
                          ('%s: ' % e['user']) if e.get('user') else '', e['text']) for e in data))
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())


# ─────────────────────────── главное окно ───────────────────────────

# ─────────────────────────── вкладка «Пульт» ───────────────────────────

class FaceWidget(QWidget):
    """Лицо Кузьмича. Меняет выражение по состоянию: спокоен / едет / нет токена."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(220, 200)
        self.state = 'idle'   # idle | move | locked

    def set_state(self, s):
        if s != self.state:
            self.state = s
            self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        r = min(w, h) * 0.42
        cx, cy = w / 2, h / 2
        if self.state == 'locked':
            face = QColor('#9aa4b0')
        elif self.state == 'move':
            face = QColor('#2f56b0')
        else:
            face = QColor('#3a5fc8')
        p.setPen(QPen(QColor('#1c2f6b'), 3))
        p.setBrush(QBrush(face))
        p.drawEllipse(QRectF(cx - r, cy - r * 0.8, r * 2, r * 1.6))
        # глаза
        p.setBrush(QBrush(QColor('#16224a')))
        eye = r * 0.13
        ey = cy - r * 0.15
        p.drawEllipse(QRectF(cx - r * 0.42 - eye, ey - eye, eye * 2, eye * 2))
        p.drawEllipse(QRectF(cx + r * 0.42 - eye, ey - eye, eye * 2, eye * 2))
        # рот
        pen = QPen(QColor('#16224a'), 4)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        mw = r * 0.9
        my = cy + r * 0.30
        if self.state == 'locked':
            p.drawLine(int(cx - mw / 2), int(my), int(cx + mw / 2), int(my))
        elif self.state == 'move':
            p.drawEllipse(QRectF(cx - mw * 0.28, my - mw * 0.28, mw * 0.56, mw * 0.56))  # «о»
        else:
            p.drawArc(QRectF(cx - mw / 2, my - mw * 0.6, mw, mw), 200 * 16, 140 * 16)  # улыбка
        p.end()


class OperatorTab(QWidget):
    """Пульт оператора по макету:
        слева вверху  — камера робота (включается кнопкой, не держит устройство)
        слева внизу   — живой журнал происходящего на роботе
        центр вверху  — управление движением
        справа        — речь: свой текст и заготовленные фразы

    Кнопки движения активны только пока у оператора токен движения.
    """

    ARROWS = {
        'turn_left': '↶', 'forward': '↑', 'turn_right': '↷',
        'left': '←', 'right': '→', 'back': '↓',
    }

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.actions = {}
        self.can_move = False
        self.can_speak = False
        self.cam_on = False
        self._quick_btns = []
        self._phrase_btns = []

        root = QGridLayout(self)
        root.setColumnStretch(0, 4)
        root.setColumnStretch(1, 3)
        root.setColumnStretch(2, 4)
        root.setRowStretch(0, 3)
        root.setRowStretch(1, 2)

        # ── слева вверху: камеры (панели строятся по конфигу демона) ──
        box_cam = QGroupBox('Камеры робота')
        cv = QVBoxLayout(box_cam)
        self.cam_row = QHBoxLayout()
        cv.addLayout(self.cam_row, 1)
        cv.addWidget(QLabel('Камеры включаются только по кнопке и освобождаются\n'
                            'при выключении — иначе они заблокируют модули.'))
        self.cams = {}          # id -> {'video','btn','lbl'}
        root.addWidget(box_cam, 0, 0)

        # ── слева внизу: живой журнал ──
        box_log = QGroupBox('Что происходит на роботе')
        lv = QVBoxLayout(box_log)
        self.log = QListWidget()
        self.log.setStyleSheet('font-family:Menlo,Consolas,monospace;font-size:12px;')
        lv.addWidget(self.log)
        root.addWidget(box_log, 1, 0)

        # ── центр: управление движением ──
        box_move = QGroupBox('Управление движением')
        mv = QVBoxLayout(box_move)
        self.btn_motion = QPushButton('Взять управление движением')
        self.btn_motion.setCheckable(True)
        self.btn_motion.clicked.connect(self.do_motion)
        mv.addWidget(self.btn_motion)

        self.banner = QLabel('')
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setWordWrap(True)
        mv.addWidget(self.banner)

        pad = QGridLayout()
        pad.setSpacing(6)
        self.dpad = {}
        for act, r, c in [('turn_left', 0, 0), ('forward', 0, 1), ('turn_right', 0, 2),
                          ('left', 1, 0), ('back', 1, 1), ('right', 1, 2)]:
            b = QPushButton(self.ARROWS[act])
            b.setFont(QFont('', 20))
            b.setMinimumSize(64, 54)
            b.clicked.connect(lambda _=False, a=act: self.do_action(a))
            self.dpad[act] = b
            pad.addWidget(b, r, c)
        mv.addLayout(pad)

        self.box_quick = QGroupBox('Быстрые команды')
        self.quick_grid = QGridLayout(self.box_quick)
        mv.addWidget(self.box_quick)

        self.btn_stop = QPushButton('СТОП')
        self.btn_stop.setMinimumHeight(48)
        self.btn_stop.setStyleSheet(
            'QPushButton{background:#c62828;color:white;font-size:19px;font-weight:bold;border-radius:8px;}'
            'QPushButton:disabled{background:#e0a0a0;}')
        self.btn_stop.clicked.connect(self.do_stop)
        mv.addWidget(self.btn_stop)
        mv.addStretch()
        root.addWidget(box_move, 0, 1, 2, 1)

        # ── справа: речь ──
        box_say = QGroupBox('Речь робота')
        sv = QVBoxLayout(box_say)
        sv.addWidget(QLabel('Робот скажет:'))
        say_row = QHBoxLayout()
        self.say_input = QLineEdit()
        self.say_input.setPlaceholderText('введите текст и нажмите «Сказать»')
        self.say_input.returnPressed.connect(self.do_say)
        say_row.addWidget(self.say_input, 1)
        self.btn_say = QPushButton('Сказать')
        self.btn_say.clicked.connect(self.do_say)
        say_row.addWidget(self.btn_say)
        sv.addLayout(say_row)

        # Громкость в децибелах. Отдельной строкой, а не в ряду с полем ввода:
        # ряд и так тесный, а число тут задаётся редко — один раз подобрал и
        # забыл.
        voice_row = QHBoxLayout()
        voice_row.addWidget(QLabel('Голос:'))
        self.voice = QComboBox()
        self.voice.setToolTip('Голоса двух движков, оба стоят на роботе.\n'
                              'RHVoice — тот же, что в телеопе.')
        voice_row.addWidget(self.voice, 1)
        sv.addLayout(voice_row)

        # Ползунок, а не поле: громкость подбирают на слух, двигая туда-сюда,
        # и цифру при этом набирать неудобно. Как в телеопе.
        gain_row = QHBoxLayout()
        gain_row.addWidget(QLabel('Усиление:'))
        self.gain = QSlider(Qt.Horizontal)
        self.gain.setRange(-20, 24)
        self.gain.setValue(0)
        gain_row.addWidget(self.gain, 1)
        self.lbl_gain_val = QLabel('0 дБ')
        self.lbl_gain_val.setMinimumWidth(52)
        gain_row.addWidget(self.lbl_gain_val)
        sv.addLayout(gain_row)

        self.lbl_gain = QLabel('')
        self.lbl_gain.setStyleSheet('color:#6c7682;font-size:11px;')
        sv.addWidget(self.lbl_gain)
        self.gain.valueChanged.connect(self.on_gain_changed)

        self.lbl_say = QLabel('')
        self.lbl_say.setWordWrap(True)
        self.lbl_say.setStyleSheet('color:#6c7682')
        sv.addWidget(self.lbl_say)

        sv.addWidget(QLabel('Заготовленные фразы:'))
        # Фраз много и они сгруппированы — без прокрутки панель не помещается
        self.phrase_scroll = QScrollArea()
        self.phrase_scroll.setWidgetResizable(True)
        self.phrase_scroll.setFrameShape(QScrollArea.NoFrame)
        self.phrase_box = QWidget()
        self.phrase_v = QVBoxLayout(self.phrase_box)
        self.phrase_v.setContentsMargins(0, 0, 6, 0)
        self.phrase_v.setSpacing(3)
        self.phrase_scroll.setWidget(self.phrase_box)
        sv.addWidget(self.phrase_scroll, 1)
        root.addWidget(box_say, 0, 2, 2, 1)

        self.set_can_move(False)
        self.set_can_speak(False)

    # ── настройка по данным демона ──
    def configure(self, move_desc, speech_desc, cameras):
        self.actions = move_desc.get('actions', {})
        for act, b in self.dpad.items():
            a = self.actions.get(act)
            if a:
                b.setToolTip('%s %s (по умолчанию %s, макс %s)'
                             % (a['cmd'], a.get('unit', ''), a.get('default', '?'), a.get('max', '?')))
            b.setVisible(act in self.actions)

        # быстрые команды. Пересобираем с нуля: configure() зовётся на каждый
        # welcome, а Link переподключается сам — иначе список рос бы и
        # set_can_move() лез в уже удалённые виджеты.
        while self.quick_grid.count():
            it = self.quick_grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self._quick_btns = []
        for i, q in enumerate(move_desc.get('quick', [])):
            b = QPushButton(q.get('label', '?'))
            b.clicked.connect(lambda _=False, a=q['action'], v=q.get('value'): self.do_action(a, v))
            self.quick_grid.addWidget(b, i // 2, i % 2)
            self._quick_btns.append(b)

        if move_desc.get('dry_run'):
            self.banner.setText('⚠ Тестовый режим: команды НЕ исполняются на роботе')
            self.banner.setStyleSheet('background:#fcedde;color:#8a5a00;padding:6px;border-radius:6px;')
        else:
            self.banner.setText('● Боевой режим: кнопки реально двигают робота')
            self.banner.setStyleSheet('background:#f8e3e3;color:#c62828;padding:6px;border-radius:6px;font-weight:bold;')

        # фразы
        while self.phrase_v.count():
            it = self.phrase_v.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self._phrase_btns = []
        groups = speech_desc.get('phrase_groups')
        if not groups:
            flat = speech_desc.get('phrases', [])
            groups = [{'title': '', 'phrases': flat}] if flat else []
        for g in groups:
            title = g.get('title', '')
            if title:
                lab = QLabel(title)
                lab.setStyleSheet('color:#6c7682;font-size:11px;font-weight:bold;'
                                  'text-transform:uppercase;padding:6px 0 2px 0;')
                self.phrase_v.addWidget(lab)
            for ph in g.get('phrases', []):
                b = QPushButton(ph)
                b.setStyleSheet('text-align:left;padding:5px 8px;')
                b.setToolTip(ph)
                b.clicked.connect(lambda _=False, t=ph: self.do_say(t))
                self.phrase_v.addWidget(b)
                self._phrase_btns.append(b)
        self.phrase_v.addStretch()
        self.say_input.setMaxLength(int(speech_desc.get('max_chars', 300)))
        self.gain.setRange(int(speech_desc.get('gain_db_min', -20)),
                           int(speech_desc.get('gain_db_max', 24)))
        self.gain.setValue(int(speech_desc.get('gain_db', 0)))
        self.on_gain_changed(self.gain.value())

        # Список голосов приходит от демона: добавить голос = дописать строку
        # в speech_commands.json на роботе, клиент менять не нужно.
        self.voice.clear()
        for v in (speech_desc.get('voices') or []):
            self.voice.addItem(v.get('title', v.get('id', '?')), v.get('id'))
        cur = speech_desc.get('voice')
        if cur:
            i = self.voice.findData(cur)
            if i >= 0:
                self.voice.setCurrentIndex(i)
        self.voice.setEnabled(self.voice.count() > 1)

        self.build_cameras(cameras or [])

    # ── состояния ──
    def set_can_move(self, ok):
        self.can_move = ok
        for b in list(self.dpad.values()) + self._quick_btns:
            b.setEnabled(ok)
        self.btn_stop.setEnabled(ok)
        self.box_quick.setEnabled(ok)

    def on_gain_changed(self, v):
        self.lbl_gain_val.setText('%+d дБ' % v if v else '0 дБ')
        # Предупреждаем ДО нажатия «Сказать», а не после того, как робот
        # прохрипел фразу на весь зал.
        if v > 6:
            self.lbl_gain.setText('выше +6 дБ вероятен хрип')
            self.lbl_gain.setStyleSheet('color:#ef6c00;font-size:11px;')
        elif v < -10:
            self.lbl_gain.setText('очень тихо')
            self.lbl_gain.setStyleSheet('color:#6c7682;font-size:11px;')
        else:
            self.lbl_gain.setText('')

    def set_can_speak(self, ok):
        self.can_speak = ok
        self.say_input.setEnabled(ok)
        self.btn_say.setEnabled(ok)
        self.voice.setEnabled(ok and self.voice.count() > 1)
        self.gain.setEnabled(ok)
        for b in self._phrase_btns:
            b.setEnabled(ok)

    def build_cameras(self, cameras):
        """Панели камер строятся по списку от демона: добавить третью камеру
        можно правкой cameras.json, без изменений в GUI."""
        while self.cam_row.count():
            it = self.cam_row.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.cams = {}
        if not cameras:
            cameras = [{'id': 'depth', 'title': 'Камера'}]
        for c in cameras:
            box = QWidget()
            v = QVBoxLayout(box)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(4)
            lab = QLabel(c.get('title', c['id']))
            lab.setStyleSheet('font-weight:bold;')
            v.addWidget(lab)
            video = QLabel('выключена')
            video.setAlignment(Qt.AlignCenter)
            video.setMinimumSize(220, 180)
            video.setStyleSheet('background:#11151a;color:#7c8b9c;border-radius:6px;')
            v.addWidget(video, 1)
            btn = QPushButton('Включить')
            btn.setCheckable(True)
            btn.clicked.connect(lambda _=False, cid=c['id']: self.do_camera(cid))
            v.addWidget(btn)
            st = QLabel('выключена')
            st.setStyleSheet('color:#6c7682;font-size:11px;')
            v.addWidget(st)
            self.cam_row.addWidget(box, 1)
            self.cams[c['id']] = {'video': video, 'btn': btn, 'lbl': st}
            self.set_camera_state(c['id'], 'on' if c.get('on') else 'off', owner=c.get('owner'))

    def set_camera_state(self, cam_id, state, owner=None, msg=None):
        c = self.cams.get(cam_id)
        if not c:
            return
        if state == 'on':
            c['btn'].setChecked(True)
            c['btn'].setText('Выключить')
            c['lbl'].setText('включена' + (' (%s)' % owner if owner and owner != self.win.user else ''))
            c['lbl'].setStyleSheet('color:#2e7d32;font-size:11px;')
        elif state == 'error':
            c['btn'].setChecked(False)
            c['btn'].setText('Включить')
            c['lbl'].setText('ошибка')
            c['lbl'].setStyleSheet('color:#c62828;font-size:11px;')
            c['video'].setPixmap(QPixmap())
            c['video'].setText('недоступна')
            if msg:
                QMessageBox.warning(self, 'Камера', msg)
        else:
            c['btn'].setChecked(False)
            c['btn'].setText('Включить')
            c['lbl'].setText('выключена')
            c['lbl'].setStyleSheet('color:#6c7682;font-size:11px;')
            c['video'].setPixmap(QPixmap())
            c['video'].setText('выключена')

    # ── действия ──
    def do_motion(self):
        self.win.link.send({'op': 'motion.acquire' if self.btn_motion.isChecked() else 'motion.release'})

    def do_action(self, action, value=None):
        if not self.can_move:
            return
        msg = {'op': 'move', 'action': action}
        if value is not None:
            msg['value'] = value
        self.win.link.send(msg)

    def do_stop(self):
        self.win.link.send({'op': 'move.stop'})

    def do_camera(self, cam_id):
        c = self.cams.get(cam_id)
        on = c['btn'].isChecked() if c else True
        self.win.link.send({'op': 'camera.start' if on else 'camera.stop', 'id': cam_id})

    def do_say(self, text=None):
        if not self.can_speak:
            return
        if not isinstance(text, str) or not text:
            text = self.say_input.text().strip()
        if not text:
            return
        self.win.link.send({'op': 'speech.say', 'text': text,
                            'gain_db': self.gain.value(),
                            'voice': self.voice.currentData()})
        if text == self.say_input.text().strip():
            self.say_input.clear()

    # ── приём от демона ──
    def on_camera(self, msg):
        cam_id = msg.get('id') or (next(iter(self.cams)) if self.cams else None)
        c = self.cams.get(cam_id)
        st = msg.get('state')
        if st == 'frame':
            if not c:
                return
            try:
                import base64
                pm = QPixmap()
                pm.loadFromData(base64.b64decode(msg['jpeg']), 'JPEG')
                if not pm.isNull():
                    v = c['video']
                    v.setPixmap(pm.scaled(v.width(), v.height(),
                                          Qt.KeepAspectRatio, Qt.SmoothTransformation))
            except Exception:
                pass
        else:
            self.set_camera_state(cam_id, st, owner=msg.get('owner'), msg=msg.get('msg'))

    def on_move(self, msg):
        if msg.get('msg'):
            self.add_log('⚠ ' + msg['msg'], COL_CRIT)
        elif msg.get('line'):
            self.add_log(msg['line'], COL_OK if msg.get('ok', True) else COL_CRIT)

    def on_speech(self, msg):
        if msg.get('busy'):
            self.lbl_say.setText('говорит: «%s»' % msg.get('text', ''))
            self.lbl_say.setStyleSheet('color:#2e7d32')
        else:
            self.lbl_say.setText(msg.get('msg', ''))
            self.lbl_say.setStyleSheet('color:#6c7682' if msg.get('ok') else 'color:#c62828')

    def on_log(self, records):
        for r in records:
            self.add_log('%s%s' % (('%s: ' % r['user']) if r.get('user') else '', r['text']),
                         stamp=r.get('t'))

    def add_log(self, text, color=None, stamp=None):
        ts = time.strftime('%H:%M:%S', time.localtime(stamp)) if stamp else time.strftime('%H:%M:%S')
        it = QListWidgetItem('%s  %s' % (ts, text))
        if color:
            it.setForeground(QBrush(color))
        self.log.addItem(it)
        self.log.scrollToBottom()
        while self.log.count() > 300:
            self.log.takeItem(0)
class PanelTab(QWidget):
    """Веб-панель робота живой страницей: карта лидара, джойстик рук и т.д.

    Встраиваем сайт как есть (QtWebEngine), а не картинку: у карты свои зум и
    режимы, у джойстика — кнопки, которые надо удерживать. Ни то, ни другое
    картинкой не заменить.

    Адрес берём ОТ ТОГО ЖЕ ХОСТА, к которому подключена консоль. Это важно при
    работе через SSH-туннель: там консоль идёт на 127.0.0.1, и панель должна
    идти туда же — иначе окно будет пустым при живой связи. Порт тоже надо
    пробросить, об этом написано в подписи под адресом.
    """

    def __init__(self, desc, host):
        super().__init__()
        self.desc = desc
        self.url = 'http://%s:%s%s' % (host, desc.get('port', 80), desc.get('page_path', '/'))
        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        self.lbl_url = QLabel(self.url)
        self.lbl_url.setStyleSheet('color:#6c7682')
        self.lbl_url.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bar.addWidget(self.lbl_url, 1)
        mod = desc.get('requires_module')
        if mod:
            hint = QLabel('нужен модуль: %s' % mod)
            hint.setStyleSheet('color:#6c7682;font-size:11px;')
            bar.addWidget(hint)
        self.btn_reload = QPushButton('Обновить')
        self.btn_reload.clicked.connect(self.reload)
        bar.addWidget(self.btn_reload)
        root.addLayout(bar)

        self.view = None
        self.hint = QLabel('Загружаю…')
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet('background:#11151a;color:#7c8b9c;border-radius:6px;padding:20px;')
        root.addWidget(self.hint, 1)

        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
        except Exception as e:
            # На чужом ноутбуке модуля может не быть — не падаем, а даём адрес,
            # чтобы панель открыли браузером.
            self.hint.setText('Встроенный просмотр недоступен (%s).\n\n'
                              'Откройте в браузере:\n%s' % (type(e).__name__, self.url))
            return
        self.view = QWebEngineView()
        root.replaceWidget(self.hint, self.view)
        self.hint.hide()
        self.reload()

    def reload(self):
        if self.view is not None:
            from PySide6.QtCore import QUrl
            self.view.setUrl(QUrl(self.url))


class MainWindow(QMainWindow):
    def __init__(self, link, user):
        super().__init__()
        self.link = link
        self.user = user
        self.level = 0
        self.setWindowTitle('Консоль G1 — %s' % user)
        self.resize(1400, 860)

        self.tabs = QTabWidget()
        self.bios = BiosTab()
        self.operator = OperatorTab(self)
        self.mods = ModulesTab(self)
        self.space = MySpaceTab(self)
        self.admin = AdminTab(self)
        self.tabs.addTab(self.bios, 'BIOS')
        self.tabs.addTab(self.operator, 'Пульт')
        self.tabs.addTab(self.mods, 'Модули')
        self.tabs.addTab(self.space, 'Моё пространство')
        self.panels = {}          # id -> PanelTab, строятся по welcome
        self.setCentralWidget(self.tabs)

        self.setStatusBar(QStatusBar())
        self.lbl_conn = QLabel('—')
        self.lbl_level = QLabel('')
        self.lbl_motion = QLabel('')
        for w in (self.lbl_conn, self.lbl_level, self.lbl_motion):
            self.statusBar().addPermanentWidget(w)

        link.message.connect(self.on_message)
        link.status.connect(self.on_status)

        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.link.send({'op': 'ping'}))
        self.timer.start(15000)

    def on_status(self, text, connected):
        self.lbl_conn.setText(text)
        self.lbl_conn.setStyleSheet('color:#2e7d32' if connected else 'color:#c62828')

    def on_message(self, msg):
        op = msg.get('op')
        if op == 'welcome':
            self.level = msg['level']
            self.lbl_level.setText('%s · %s · клиент v%d' % (msg['user'], msg['level_name'],
                                                              CLIENT_VERSION))
            need = int(msg.get('client_min') or 0)
            if need > CLIENT_VERSION:
                self.lbl_level.setStyleSheet('color:#c62828;font-weight:bold')
                self.offer_update(need)
            self.on_status('подключено', True)
            self.bios.configure(msg.get('config', {}))
            self.operator.configure(msg.get('move', {}), msg.get('speech', {}), msg.get('cameras', []))
            self.build_panels(msg.get('panels') or [])
            self.link.send({'op': 'log.sub', 'on': True})   # живой журнал в «Пульт»
            self.space.lbl_dir.setText('личный каталог: %s' % msg.get('user_dir', '—'))
            if self.level >= 5 and self.tabs.indexOf(self.admin) < 0:
                self.tabs.addTab(self.admin, 'Админ')
            self.set_motion(msg.get('motion'))
            perms = msg.get('perms', {})
            self.operator.btn_motion.setEnabled(perms.get('motion.acquire', False))
            self.operator.set_can_speak(perms.get('module.start', False))
            self.mods.btn_motion.setEnabled(perms.get('motion.acquire', False))
            self.mods.btn_start.setEnabled(perms.get('module.start', False))
            self.mods.btn_stop.setEnabled(perms.get('module.stop', False))
            self.space.btn_add.setEnabled(perms.get('module.register', False))
            if not perms.get('module.register', False):
                self.space.setEnabled(False)
        elif op == 'telemetry':
            self.bios.update_data(msg['data'], msg.get('alerts', []))
        elif op == 'modules':
            self.mods.set_modules(msg['data'], msg.get('running', []))
        elif op == 'logs':
            self.mods.show_logs(msg['key'], msg['data'])
        elif op == 'motion':
            self.set_motion(msg.get('data'))
        elif op == 'move':
            self.operator.on_move(msg)
        elif op == 'camera':
            self.operator.on_camera(msg)
        elif op == 'speech':
            self.operator.on_speech(msg)
        elif op == 'log.line':
            self.operator.on_log([msg['data']])
        elif op == 'sessions':
            self.admin.set_sessions(msg['data'])
        elif op == 'log':
            self.admin.set_log(msg['data'])
            self.operator.on_log(msg['data'])   # первичное заполнение журнала пульта
        elif op == 'client.file':
            self.apply_update(msg)
        elif op == 'error':
            self.statusBar().showMessage('Ошибка: %s' % msg['msg'], 8000)
            QMessageBox.warning(self, 'Отказано', msg['msg'])
        elif op == 'ok':
            self.statusBar().showMessage('Готово: %s' % (msg.get('ref') or ''), 4000)

    def build_panels(self, panels):
        """Вкладки панелей появляются по списку от демона.

        Строим ОДИН раз: welcome приходит на каждое переподключение, а
        пересоздание QWebEngineView каждый раз рвало бы открытую страницу и
        сбрасывало то, что оператор на ней настроил.
        """
        for p in panels:
            pid = p.get('id')
            if not pid or pid in self.panels:
                continue
            # host именно тот, к которому подключились: через SSH-туннель это
            # 127.0.0.1, и панель должна идти тем же путём.
            tab = PanelTab(p, self.link.host)
            self.panels[pid] = tab
            self.tabs.addTab(tab, p.get('title', pid))

    def offer_update(self, need):
        """Предложить обновиться. Именно предложить, а не сделать молча.

        Это клиент, который скачивает КОД с сервера по открытому TCP без TLS.
        Тот, кто сможет притвориться демоном в сети робота, положит оператору на
        ноутбук что угодно. Поэтому обновление всегда с явного согласия, с
        показом источника, и с резервной копией — чтобы откат был в одну команду.
        """
        r = QMessageBox.question(
            self, 'Клиент устарел',
            'Ваш g1_console_gui.py версии %d, роботу нужен %d.\n'
            'Часть вкладок не появится.\n\n'
            'Скачать свежий файл с %s:%d и заменить текущий?\n'
            'Старый сохранится рядом с расширением .bak\n\n'
            'Это загрузка кода с сервера по незашифрованному каналу — '
            'соглашайтесь, только если доверяете сети робота.'
            % (CLIENT_VERSION, need, self.link.host, self.link.port),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if r == QMessageBox.Yes:
            self.statusBar().showMessage('качаю свежий клиент…', 30000)
            self.link.send({'op': 'client.download'})

    def apply_update(self, msg):
        import base64, hashlib, os, shutil
        try:
            raw = base64.b64decode(msg['b64'])
        except Exception as e:
            return QMessageBox.warning(self, 'Обновление', 'Файл не разобрался: %r' % e)

        # Три проверки перед заменой рабочего файла. Записать мусор поверх
        # единственного клиента — значит оставить оператора вообще без консоли.
        got = hashlib.sha256(raw).hexdigest()
        if msg.get('sha256') and got != msg['sha256']:
            return QMessageBox.warning(self, 'Обновление',
                                       'Контрольная сумма не совпала — файл повреждён при передаче.')
        if b'CLIENT_VERSION' not in raw or b'class MainWindow' not in raw:
            return QMessageBox.warning(self, 'Обновление',
                                       'Пришло не похоже на клиент консоли — замена отменена.')
        try:
            compile(raw, 'g1_console_gui.py', 'exec')
        except SyntaxError as e:
            return QMessageBox.warning(self, 'Обновление',
                                       'Пришедший файл не компилируется: %s' % e)

        me = os.path.abspath(__file__)
        try:
            shutil.copy2(me, me + '.bak')
            with open(me, 'wb') as fh:
                fh.write(raw)
        except Exception as e:
            return QMessageBox.warning(self, 'Обновление',
                                       'Не удалось записать файл: %r\n\n'
                                       'Возможно, нет прав на каталог.' % e)
        QMessageBox.information(
            self, 'Готово',
            'Клиент обновлён до версии %s (%d КБ).\n'
            'Старый сохранён: %s.bak\n\n'
            'Закройте окно и запустите консоль заново.'
            % (msg.get('version', '?'), len(raw) // 1024, os.path.basename(me)))

    def set_motion(self, motion):
        if motion:
            mine = motion['user'] == self.user
            self.lbl_motion.setText('движение: %s%s' % (motion['user'], ' (вы)' if mine else ''))
            self.lbl_motion.setStyleSheet('color:#2e7d32' if mine else 'color:#ef6c00')
            self.mods.btn_motion.setChecked(mine)
            self.mods.btn_motion.setText('Отдать управление' if mine else 'Токен у %s' % motion['user'])
            self.operator.btn_motion.setChecked(mine)
            self.operator.btn_motion.setText('Отдать управление' if mine else 'Токен у %s' % motion['user'])
            self.operator.set_can_move(mine)
        else:
            self.lbl_motion.setText('движение: свободно')
            self.lbl_motion.setStyleSheet('color:#9e9e9e')
            self.mods.btn_motion.setChecked(False)
            self.mods.btn_motion.setText('Взять управление движением')
            self.operator.btn_motion.setChecked(False)
            self.operator.btn_motion.setText('Взять управление движением')
            self.operator.set_can_move(False)

    def closeEvent(self, e):
        self.link.close()
        e.accept()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default=os.environ.get('G1_HOST', '127.0.0.1'))
    ap.add_argument('--port', type=int, default=15100)
    ap.add_argument('--user', default=os.environ.get('G1_USER', ''))
    ap.add_argument('--token', default=os.environ.get('G1_TOKEN', ''))
    a = ap.parse_args()

    app = QApplication(sys.argv)
    host, port, user, token = a.host, a.port, a.user, a.token
    if not user or not token:
        dlg = LoginDialog(host, port, user)
        if dlg.exec() != QDialog.Accepted:
            return
        host, port, user, token = dlg.values()

    link = Link(host, port, user, token)
    win = MainWindow(link, user)
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
