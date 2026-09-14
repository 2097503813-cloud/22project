"""独立小工具（与本平台的训练/推理链路无关）：把一个 txt 里的每一行做成一个 Word 文档。

用途是「一个标题一行 → 一份文档」的批量生成，靠 COM 驱动本机已安装的 Office 完成。

前提：Windows + 已安装 Microsoft Word + `pip install pywin32`。
跑法：把 __main__ 里的两个路径改成你自己的，然后 `python createdoc.py`。
"""

import os

import win32com.client as win32


def clean_filename(title):
    """把一行文字变成合法的 Windows 文件名（替换非法字符、去掉首尾空白与点）。"""
    invalid_chars = r'\/:*?"<>|'
    for ch in invalid_chars:
        title = title.replace(ch, '_')
    # 结尾的 '.' 在 Windows 上会被系统吞掉，这里主动去掉，免得存出来的名字对不上
    return title.strip().strip('.')


def create_docs_from_lines_win32(txt_file_path, output_dir="output_docs_win32"):
    """读 txt，每行（非空）生成一个 .doc，存到 output_dir，返回成功生成的个数。

    文件名形如 `001_标题.doc`——序号用行号补齐 3 位，保证资源管理器里按名称排序
    与 txt 里的顺序一致。
    """
    os.makedirs(output_dir, exist_ok=True)
    word = None          # 放在 try 外面：finally 里要靠它判断"Word 到底启起来没有"
    count = 0
    try:
        # 启动 Word（确保已安装 Office）。EnsureDispatch 会走早期绑定生成缓存，
        # 这样下面才能用 win32.constants.wdColorBlue 这类常量
        word = win32.gencache.EnsureDispatch('Word.Application')
        word.Visible = False  # 不显示窗口
        print(txt_file_path)

        with open(txt_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines, start=1):
            title = line.strip()
            print(title)
            if not title:                     # 跳过空行，但行号照常递增（序号与源文件对齐）
                continue

            safe_title = clean_filename(title)
            # 限制文件名长度
            if len(safe_title) > 50:
                safe_title = safe_title[:50]
            filename = f"{idx:03d}_{safe_title}.doc"  # 或 .docx
            filepath = os.path.join(output_dir, filename)

            doc = word.Documents.Add()
            # 添加标题段落
            para = doc.Paragraphs.Add()
            para.Range.Text = title
            para.Range.Font.Size = 24
            para.Range.Font.Bold = True
            para.Range.Font.Color = win32.constants.wdColorBlue
            para.Range.Font.Name = "微软雅黑"
            para.Alignment = win32.constants.wdAlignParagraphCenter
            para.Range.InsertParagraphAfter()

            # 可添加额外内容
            # doc.Content.InsertAfter("正文内容...")

            doc.SaveAs(filepath)
            doc.Close()                       # 不关掉的话，Word 里会堆着几百个隐藏文档
            count += 1
            print(f"已生成: {filepath}")

        print(f"全部完成！共生成 {count} 个文档，保存在 '{output_dir}'。")
    except Exception as e:
        # 这里刻意不往上抛：批量跑到一半失败时，前面已经生成的文档要保留，
        # 调用方看 print 出来的进度就知道断在哪一行
        print(e)
    finally:
        # 无论是否出错，都释放 Word 进程
        if word is not None:
            word.Quit()



if __name__ == '__main__':
    # 请修改为你的文本文件路径
    create_docs_from_lines_win32('E:\\zsd\\zsd.txt','E:\\zsd\\zsd')
