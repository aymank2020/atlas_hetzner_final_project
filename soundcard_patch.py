"""
رقعة مكتبة soundcard — Soundcard Library Monkey-Patch
======================================================
يصلح مشكلتين في مكتبة soundcard على ويندوز:

المشكلة الأولى: خطأ COM رقم 0x100000001
   السبب: لما ويندوز يرد بـ S_FALSE (يعني "COM متصل بالفعل")
   المكتبة بتعتبره خطأ فادح وبتقفل
   الحل: نعلّم المكتبة إن S_FALSE ده رد طبيعي مش خطأ

المشكلة الثانية: numpy.fromstring قديمة ومحذوفة
   السبب: المكتبة بتستخدم دالة اتشالت من numpy
   الحل: نستبدلها بـ numpy.frombuffer الجديدة

الاستخدام: استورد الملف ده قبل أي استخدام لـ soundcard
   import soundcard_patch
   soundcard_patch.apply()  # مرة واحدة بس
"""

import sys
import platform
import logging

logger = logging.getLogger("AtlasScribe.SoundcardPatch")

_patched = False


def apply():
    """
    طبّق الرقعة على مكتبة soundcard.
    آمنة للاستدعاء أكثر من مرة — بتشتغل مرة واحدة بس.
    """
    global _patched
    if _patched:
        return True

    if platform.system() != "Windows":
        logger.info("مش ويندوز — مفيش حاجة نصلحها")
        _patched = True
        return True

    try:
        success = _patch_com_library()
        if success:
            _patch_numpy_fromstring()
            _patched = True
            logger.info("✅ تم تطبيق رقعة soundcard بنجاح")
        return success
    except Exception as e:
        logger.error(f"❌ فشل تطبيق الرقعة: {e}")
        return False


def _patch_com_library():
    """
    إصلاح مشكلة COM S_FALSE:
    المكتبة بتستخدم check_error() اللي مش بتعرف إن S_FALSE (1)
    معناه "تمام، COM متصل بالفعل" — فبتعتبره خطأ.

    الحل: نعدّل __init__ بتاعت _COMLibrary عشان تتعامل مع S_FALSE صح.
    """
    try:
        from soundcard import mediafoundation as mf
        import cffi

        original_init = mf._COMLibrary.__init__

        def patched_init(self):
            """نسخة مصلحة من _COMLibrary.__init__ بتتعامل مع S_FALSE."""
            COINIT_MULTITHREADED = 0x0
            S_OK = 0
            S_FALSE = 1  # ← الكود اللي المكتبة مش عارفاه!
            RPC_E_CHANGED_MODE = 0x80010106

            _ole32 = mf._ole32
            _ffi = mf._ffi

            if platform.win32_ver()[0] == '8':
                hr = _ole32.CoInitialize(_ffi.NULL)
            else:
                hr = _ole32.CoInitializeEx(_ffi.NULL, COINIT_MULTITHREADED)

            if hr == S_OK:
                # COM اتفتح بنجاح — لازم نقفله بعدين
                self.com_loaded = True
                logger.debug("COM initialized successfully (S_OK)")
            elif hr == S_FALSE:
                # COM كان مفتوح بالفعل — تمام، مش هنقفله
                self.com_loaded = False
                logger.debug("COM was already initialized (S_FALSE) — OK!")
            elif (hr + 2**32) == RPC_E_CHANGED_MODE:
                # COM مفتوح بوضع مختلف — مش مشكلة
                self.com_loaded = False
                logger.debug("COM already initialized in different mode (RPC_E_CHANGED_MODE) — OK!")
            else:
                # خطأ حقيقي — نرميه
                self.com_loaded = False
                mf._COMLibrary.check_error(hr)

        # طبّق الرقعة
        mf._COMLibrary.__init__ = patched_init

        # أعد إنشاء الكائن العام _com بالكود المصلح
        try:
            mf._com = mf._COMLibrary()
            logger.info("✅ تم إصلاح COM — _com أعيد إنشاؤه بنجاح")
        except Exception as e:
            logger.warning(f"⚠️ فشل إعادة إنشاء _com: {e}")
            # حاول مرة تانية مع COM مبدئي يدوي
            import ctypes
            ctypes.windll.ole32.CoInitializeEx(None, 0)
            mf._com = mf._COMLibrary()
            logger.info("✅ تم إصلاح COM بعد تهيئة يدوية")

        return True

    except Exception as e:
        logger.error(f"❌ فشل إصلاح COM: {e}")
        import traceback
        traceback.print_exc()
        return False


def _patch_numpy_fromstring():
    """
    إصلاح مشكلة numpy.fromstring المحذوفة:
    المكتبة بتستخدم numpy.fromstring اللي اتشالت في numpy الجديدة.
    الحل: نستبدلها بـ numpy.frombuffer في دالة _record_chunk.
    """
    try:
        from soundcard import mediafoundation as mf
        import numpy

        _ffi = mf._ffi
        _ole32 = mf._ole32

        # احفظ الدالة الأصلية
        if not hasattr(mf._Recorder, '_original_record_chunk'):
            mf._Recorder._original_record_chunk = mf._Recorder._record_chunk

        def patched_record_chunk(self):
            """نسخة مصلحة من _record_chunk بتستخدم frombuffer بدل fromstring."""
            import time
            import warnings

            while self._capture_available_frames() == 0:
                if self._idle_start_time is None:
                    self._idle_start_time = time.perf_counter_ns()

                default_block_length, minimum_block_length = self.deviceperiod
                time.sleep(minimum_block_length / 4)
                elapsed_time_ns = time.perf_counter_ns() - self._idle_start_time

                if elapsed_time_ns / 1_000_000_000 > default_block_length * 4:
                    num_frames = int(self.samplerate * elapsed_time_ns / 1_000_000_000)
                    num_channels = len(set(self.channelmap))
                    self._idle_start_time += elapsed_time_ns
                    return numpy.zeros([num_frames * num_channels], dtype='float32')

            self._idle_start_time = None
            data_ptr, nframes, flags = self._capture_buffer()

            if data_ptr != _ffi.NULL:
                buf = _ffi.buffer(data_ptr, nframes * 4 * len(set(self.channelmap)))
                # ← هنا التصليح: frombuffer بدل fromstring
                chunk = numpy.frombuffer(buf, dtype='float32').copy()
            else:
                raise RuntimeError('Could not create capture buffer')

            if flags & _ole32.AUDCLNT_BUFFERFLAGS_SILENT:
                chunk[:] = 0

            if self._is_first_frame:
                flags &= ~_ole32.AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY
                self._is_first_frame = False

            if flags & _ole32.AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY:
                warnings.warn("data discontinuity in recording",
                              mf.SoundcardRuntimeWarning)

            if nframes > 0:
                self._capture_release(nframes)
                return chunk
            else:
                return numpy.zeros([0], dtype='float32')

        mf._Recorder._record_chunk = patched_record_chunk
        logger.info("✅ تم إصلاح numpy.fromstring → frombuffer")

    except Exception as e:
        logger.warning(f"⚠️ فشل إصلاح fromstring (ممكن يشتغل بدونها): {e}")


def verify():
    """
    تحقق إن الرقعة شغالة — اختبار سريع.
    """
    if not _patched:
        print("⚠️ الرقعة مش مطبقة — شغّل apply() الأول")
        return False

    try:
        import soundcard as sc
        speakers = sc.all_speakers()
        mics = sc.all_microphones(include_loopback=False)
        loopbacks = sc.all_microphones(include_loopback=True)

        print(f"✅ سماعات: {len(speakers)}")
        for s in speakers:
            print(f"   🔊 {s.name}")

        print(f"✅ مايكروفونات: {len(mics)}")
        for m in mics:
            print(f"   🎤 {m.name}")

        print(f"✅ أجهزة Loopback: {len(loopbacks) - len(mics)}")
        for lb in loopbacks:
            if lb not in mics:
                print(f"   🔁 {lb.name}")

        return True
    except Exception as e:
        print(f"❌ فشل التحقق: {e}")
        return False


# لو شغّلت الملف ده لوحده — هيطبق الرقعة ويتحقق
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    print("=" * 50)
    print("🔧 تطبيق رقعة soundcard...")
    print("=" * 50)

    if apply():
        print("\n🧪 اختبار التحقق:")
        verify()
    else:
        print("❌ فشل التطبيق!")
