import win32gui
import win32api
import win32process
import ctypes

Psapi = ctypes.WinDLL('Psapi.dll')
Kernel32 = ctypes.WinDLL('kernel32.dll')
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010

def EnumProcessModulesEx(hProcess):
    buf_count = 256
    while True:
        LIST_MODULES_ALL = 0x03
        buf = (ctypes.wintypes.HMODULE * buf_count)()
        buf_size = ctypes.sizeof(buf)
        needed = ctypes.wintypes.DWORD()
        if not Psapi.EnumProcessModulesEx(hProcess, ctypes.byref(buf), buf_size, ctypes.byref(needed), LIST_MODULES_ALL):
            raise OSError('EnumProcessModulesEx failed')
        if buf_size < needed.value:
            buf_count = needed.value // (buf_size // buf_count)
            continue
        count = needed.value // (buf_size // buf_count)
        return map(ctypes.wintypes.HMODULE, buf[:count])

class Hp_getter():
    def __init__(self):
        hd = win32gui.FindWindow(None, "Hollow Knight")
        pid = win32process.GetWindowThreadProcessId(hd)[1]
        self.process_handle = win32api.OpenProcess(0x1F0FFF, False, pid)
        self.kernal32 = ctypes.windll.LoadLibrary(r"C:\\Windows\\System32\\kernel32.dll")

        self.hx = 0
        # get dll address
        hProcess = Kernel32.OpenProcess(
        PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
        False, pid)
        hModule  = EnumProcessModulesEx(hProcess)
        for i in hModule:
          temp = win32process.GetModuleFileNameEx(self.process_handle,i.value)
          if temp[-15:] == "UnityPlayer.dll":
            self.UnityPlayer = i.value
          if temp[-18:] == "mono-2.0-bdwgc.dll":
            self.mono = i.value

    def get_souls(self):
        # 确保参数类型已注册（只需在初始化执行一次即可，这里重复声明以防万一）
        self.kernal32.ReadProcessMemory.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
        ]

        base_address = self.UnityPlayer + 0x019D7A48
        current_addr = ctypes.c_uint64(0)
        ptr_size = 8  # 64位指针大小

        # 前 5 个偏移用于追踪指针
        offsets = [0xA0, 0x30, 0x18, 0x60, 0xD0]
        # 最后一个偏移用于获取数值
        final_offset = 0x1CC

        # 1. 读取基地址指向的第一个指针
        self.kernal32.ReadProcessMemory(int(self.process_handle), base_address, ctypes.byref(current_addr), ptr_size,
                                        None)

        # 2. 遍历追踪指针链
        for offset in offsets:
            if current_addr.value == 0: return 0
            self.kernal32.ReadProcessMemory(int(self.process_handle), current_addr.value + offset,
                                            ctypes.byref(current_addr), ptr_size, None)

        # 3. 读取最终的灵魂数值 (4字节 int)
        soul_val = ctypes.c_int32(0)
        self.kernal32.ReadProcessMemory(int(self.process_handle), current_addr.value + final_offset,
                                        ctypes.byref(soul_val), 4, None)

        return soul_val.value

    def get_self_hp(self):
        # 基地址计算
        base_addr_val = self.mono + 0x004A7420
        current_addr = ctypes.c_uint64(0)
        ptr_size = 8

        # 前 5 个偏移
        offsets = [0x290, 0xE20, 0x0, 0x60, 0x0]
        # 最后一个偏移
        final_offset = 0x190

        # 1. 读取基地址
        res = self.kernal32.ReadProcessMemory(
            int(self.process_handle),
            ctypes.c_void_p(base_addr_val),
            ctypes.byref(current_addr),
            ptr_size,
            None
        )
        if res == 0: return -1

        # 2. 循环读取偏移
        for offset in offsets:
            if current_addr.value == 0: return 0
            self.kernal32.ReadProcessMemory(
                int(self.process_handle),
                ctypes.c_void_p(current_addr.value + offset),
                ctypes.byref(current_addr),
                ptr_size,
                None
            )

        # 3. 读取最终血量值 (4字节 int)
        hp_val = ctypes.c_int32(0)
        self.kernal32.ReadProcessMemory(
            int(self.process_handle),
            ctypes.c_void_p(current_addr.value + final_offset),
            ctypes.byref(hp_val),
            4,
            None
        )

        return hp_val.value


    def get_boss_hp(self):
        # 1. 明确声明 ReadProcessMemory 的参数类型，防止溢出
        self.kernal32.ReadProcessMemory.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
        ]
        base_address = self.UnityPlayer + 0x019D4478

        # 使用 c_uint64 来存储 64 位地址，避免溢出
        current_address = ctypes.c_uint64(0)

        # 偏移列表 (前 5 个是寻找地址)
        offsets = [0x88,0X238,0X130,0xD8,0X28]
        # 最后一个是寻找数值
        final_offset = 0x140

        try:
            # 读取第一个基地址指向的值
            self.kernal32.ReadProcessMemory(int(self.process_handle), base_address, ctypes.byref(current_address), 8,
                                            None)

            # 遍历前面的偏移量来追踪指针链
            for offset in offsets:
                self.kernal32.ReadProcessMemory(int(self.process_handle), current_address.value + offset,
                                                ctypes.byref(current_address), 8, None)

            # 此时 current_address.value 是指向 BossHP 结构体的最后基址
            # 读取最终的 HP 数值 (4 字节)
            boss_hp_value = ctypes.c_int32(0)
            self.kernal32.ReadProcessMemory(int(self.process_handle), current_address.value + final_offset,
                                            ctypes.byref(boss_hp_value), 4, None)

            res = boss_hp_value.value

            # 逻辑判断
            if res > 900:
                return 901
            elif res < 0:
                return -1
            return res

        except Exception as e:
            print(f"读取Boss血量失败: {e}")
            return -1

    # the methods below can not work yet
    def get_play_location(self):
        # 确保参数类型已注册
        self.kernal32.ReadProcessMemory.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
        ]

        base_address = self.UnityPlayer + 0x01A1DDD8
        current_addr = ctypes.c_uint64(0)
        ptr_size = 8  # 64位环境

        # 偏移量追踪
        offsets = [0xA90, 0x640, 0x140, 0x88, 0x20]
        final_offset = 0x2C

        # 1. 读取第一个指针
        self.kernal32.ReadProcessMemory(int(self.process_handle), base_address, ctypes.byref(current_addr), ptr_size,
                                        None)

        # 2. 遍历偏移列表
        for offset in offsets:
            if current_addr.value == 0: return 0.0
            self.kernal32.ReadProcessMemory(int(self.process_handle), current_addr.value + offset,
                                            ctypes.byref(current_addr), ptr_size, None)

        # 3. 读取最终的坐标数值 (4字节 float)
        pos_x = ctypes.c_float(0)
        self.kernal32.ReadProcessMemory(int(self.process_handle), current_addr.value + final_offset,
                                        ctypes.byref(pos_x), 4, None)

        return pos_x.value

    def get_hornet_location(self):
        # 确保参数类型已注册
        self.kernal32.ReadProcessMemory.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
        ]

        base_address = self.UnityPlayer + 0x019D4478
        current_addr = ctypes.c_uint64(0)
        ptr_size = 8

        # 偏移量追踪
        offsets = [0x88, 0x238, 0x50, 0x28, 0x50]
        final_offset = 0x70

        # 1. 读取第一个指针
        self.kernal32.ReadProcessMemory(int(self.process_handle), base_address, ctypes.byref(current_addr), ptr_size,
                                        None)

        # 2. 遍历偏移列表
        for offset in offsets:
            if current_addr.value == 0: return getattr(self, 'hx', 0.0)
            self.kernal32.ReadProcessMemory(int(self.process_handle), current_addr.value + offset,
                                            ctypes.byref(current_addr), ptr_size, None)

        # 3. 读取小姐姐的坐标数值 (4字节 float)
        h_pos_x = ctypes.c_float(0)
        self.kernal32.ReadProcessMemory(int(self.process_handle), current_addr.value + final_offset,
                                        ctypes.byref(h_pos_x), 4, None)

        # 逻辑判断并存储
        if 14 < h_pos_x.value < 40:
            self.hx = h_pos_x.value

        return getattr(self, 'hx', h_pos_x.value)