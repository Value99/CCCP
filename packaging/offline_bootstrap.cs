using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CCCPOfflineBootstrap
{
    [DataContract]
    internal sealed class PartInfo
    {
        [DataMember(Name = "name")] public string Name;
        [DataMember(Name = "size_bytes")] public long SizeBytes;
        [DataMember(Name = "sha256")] public string Sha256;
    }

    [DataContract]
    internal sealed class PackageInfo
    {
        [DataMember(Name = "version")] public string Version;
        [DataMember(Name = "archive_name")] public string ArchiveName;
        [DataMember(Name = "archive_size_bytes")] public long ArchiveSizeBytes;
        [DataMember(Name = "archive_sha256")] public string ArchiveSha256;
        [DataMember(Name = "root_directory")] public string RootDirectory;
        [DataMember(Name = "entry_count")] public int EntryCount;
        [DataMember(Name = "parts")] public List<PartInfo> Parts;
    }

    internal sealed class InstallUpdate
    {
        public int Percent;
        public string Stage;
        public string Detail;
    }

    internal sealed class InstallerForm : Form
    {
        private readonly Label stageLabel = new Label();
        private readonly Label detailLabel = new Label();
        private readonly ProgressBar progressBar = new ProgressBar();
        private readonly Label percentLabel = new Label();
        private readonly Button closeButton = new Button();
        private bool installing = true;

        public InstallerForm()
        {
            Text = "CCCP Launcher 完整离线版";
            Width = 620;
            Height = 250;
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            StartPosition = FormStartPosition.CenterScreen;
            BackColor = Color.FromArgb(246, 232, 201);
            ForeColor = Color.FromArgb(65, 35, 25);
            Font = new Font("Microsoft YaHei UI", 9.5f);

            Label title = new Label();
            title.Text = "CCCP 完整离线环境";
            title.Font = new Font("Microsoft YaHei UI", 16f, FontStyle.Bold);
            title.SetBounds(28, 22, 500, 34);
            Controls.Add(title);

            stageLabel.Font = new Font("Microsoft YaHei UI", 10.5f, FontStyle.Bold);
            stageLabel.Text = "正在读取发行分卷…";
            stageLabel.SetBounds(30, 69, 460, 26);
            Controls.Add(stageLabel);

            percentLabel.TextAlign = ContentAlignment.MiddleRight;
            percentLabel.Text = "0%";
            percentLabel.SetBounds(500, 69, 75, 26);
            Controls.Add(percentLabel);

            progressBar.Style = ProgressBarStyle.Continuous;
            progressBar.SetBounds(30, 101, 545, 18);
            Controls.Add(progressBar);

            detailLabel.AutoEllipsis = true;
            detailLabel.ForeColor = Color.FromArgb(116, 76, 52);
            detailLabel.Text = "不会安装 Python，也不会联网下载依赖";
            detailLabel.SetBounds(30, 130, 545, 42);
            Controls.Add(detailLabel);

            closeButton.Text = "请等待";
            closeButton.Enabled = false;
            closeButton.SetBounds(480, 174, 95, 31);
            closeButton.Click += delegate { Close(); };
            Controls.Add(closeButton);

            FormClosing += delegate(object sender, FormClosingEventArgs e) {
                if (installing && e.CloseReason == CloseReason.UserClosing) {
                    MessageBox.Show(
                        this,
                        "离线环境仍在校验或解压，请等待完成，避免留下不完整目录。",
                        "正在处理",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Information
                    );
                    e.Cancel = true;
                }
            };
            Shown += async delegate { await InstallAsync(); };
        }

        private async Task InstallAsync()
        {
            Progress<InstallUpdate> progress = new Progress<InstallUpdate>(ApplyUpdate);
            try {
                string launcher = await Task.Run(() => Install(progress));
                installing = false;
                ApplyUpdate(new InstallUpdate {
                    Percent = 100,
                    Stage = "离线环境已就绪",
                    Detail = "正在启动 CCCP-Launcher.exe"
                });
                closeButton.Text = "完成";
                closeButton.Enabled = true;
                Process.Start(new ProcessStartInfo {
                    FileName = launcher,
                    WorkingDirectory = Path.GetDirectoryName(launcher),
                    UseShellExecute = true
                });
                await Task.Delay(700);
                Close();
            }
            catch (Exception error) {
                installing = false;
                progressBar.Style = ProgressBarStyle.Continuous;
                progressBar.Value = 100;
                progressBar.ForeColor = Color.DarkRed;
                stageLabel.Text = "离线环境准备失败";
                detailLabel.Text = error.Message;
                percentLabel.Text = "失败";
                closeButton.Text = "关闭";
                closeButton.Enabled = true;
                MessageBox.Show(this, error.ToString(), "CCCP 离线版", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void ApplyUpdate(InstallUpdate update)
        {
            int value = Math.Max(0, Math.Min(100, update.Percent));
            progressBar.Value = value;
            percentLabel.Text = value.ToString() + "%";
            stageLabel.Text = update.Stage ?? "正在处理";
            detailLabel.Text = update.Detail ?? "";
        }

        private static PackageInfo ReadManifest(string baseDirectory)
        {
            string[] candidates = Directory.GetFiles(baseDirectory, "CCCP-Launcher-v*-offline.parts.json");
            if (candidates.Length != 1)
                throw new InvalidOperationException("安装器旁边必须有且只能有一份 CCCP 离线分卷清单（*.parts.json）。");
            DataContractJsonSerializer serializer = new DataContractJsonSerializer(typeof(PackageInfo));
            using (FileStream stream = File.OpenRead(candidates[0]))
                return (PackageInfo)serializer.ReadObject(stream);
        }

        private static string SafeFileName(string value, string field)
        {
            if (String.IsNullOrWhiteSpace(value) || Path.GetFileName(value) != value)
                throw new InvalidDataException(field + " 包含无效路径。");
            return value;
        }

        private static string Hex(byte[] bytes)
        {
            StringBuilder text = new StringBuilder(bytes.Length * 2);
            foreach (byte value in bytes) text.Append(value.ToString("x2"));
            return text.ToString();
        }

        private static string Install(IProgress<InstallUpdate> progress)
        {
            string baseDirectory = AppDomain.CurrentDomain.BaseDirectory;
            PackageInfo manifest = ReadManifest(baseDirectory);
            if (manifest.Parts == null || manifest.Parts.Count == 0)
                throw new InvalidDataException("离线分卷清单为空。");
            string archiveName = SafeFileName(manifest.ArchiveName, "archive_name");
            string rootName = SafeFileName(manifest.RootDirectory, "root_directory");
            string joinedArchive = Path.Combine(baseDirectory, archiveName);
            long expectedBytes = manifest.Parts.Sum(part => part.SizeBytes);
            long copiedBytes = 0;

            progress.Report(new InstallUpdate { Percent = 1, Stage = "正在校验并合并离线分卷", Detail = manifest.Parts.Count + " 个分卷" });
            using (FileStream output = new FileStream(joinedArchive, FileMode.Create, FileAccess.Write, FileShare.None, 4 * 1024 * 1024, FileOptions.SequentialScan))
            using (SHA256 archiveHash = SHA256.Create()) {
                byte[] buffer = new byte[4 * 1024 * 1024];
                foreach (PartInfo part in manifest.Parts) {
                    string partName = SafeFileName(part.Name, "part.name");
                    string partPath = Path.Combine(baseDirectory, partName);
                    if (!File.Exists(partPath)) throw new FileNotFoundException("缺少离线分卷：" + partName, partPath);
                    using (SHA256 partHash = SHA256.Create())
                    using (FileStream input = new FileStream(partPath, FileMode.Open, FileAccess.Read, FileShare.Read, buffer.Length, FileOptions.SequentialScan)) {
                        int count;
                        while ((count = input.Read(buffer, 0, buffer.Length)) > 0) {
                            output.Write(buffer, 0, count);
                            partHash.TransformBlock(buffer, 0, count, null, 0);
                            archiveHash.TransformBlock(buffer, 0, count, null, 0);
                            copiedBytes += count;
                            progress.Report(new InstallUpdate {
                                Percent = 1 + (int)(29L * copiedBytes / Math.Max(1L, expectedBytes)),
                                Stage = "正在校验并合并离线分卷",
                                Detail = partName + " · " + FormatBytes(copiedBytes) + " / " + FormatBytes(expectedBytes)
                            });
                        }
                        partHash.TransformFinalBlock(new byte[0], 0, 0);
                        if (!String.Equals(Hex(partHash.Hash), part.Sha256, StringComparison.OrdinalIgnoreCase))
                            throw new InvalidDataException("分卷校验失败，请重新下载：" + partName);
                    }
                }
                archiveHash.TransformFinalBlock(new byte[0], 0, 0);
                if (copiedBytes != manifest.ArchiveSizeBytes || !String.Equals(Hex(archiveHash.Hash), manifest.ArchiveSha256, StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException("合并后的压缩包校验失败，请重新下载缺损分卷。");
            }

            int entryCount;
            using (ZipArchive zip = ZipFile.OpenRead(joinedArchive)) {
                entryCount = zip.Entries.Count;
                foreach (ZipArchiveEntry entry in zip.Entries) {
                    string name = entry.FullName.Replace('\\', '/');
                    if (name.StartsWith("/", StringComparison.Ordinal) || name.Contains("../") || (!name.Equals(rootName, StringComparison.Ordinal) && !name.StartsWith(rootName + "/", StringComparison.Ordinal)))
                        throw new InvalidDataException("压缩包包含越界路径：" + name);
                }
            }

            progress.Report(new InstallUpdate { Percent = 31, Stage = "正在解压完整离线环境", Detail = entryCount + " 个文件和目录" });
            string tar = Path.Combine(Environment.SystemDirectory, "tar.exe");
            if (!File.Exists(tar)) throw new FileNotFoundException("系统缺少 Windows tar.exe，无法解压 ZIP。", tar);
            ProcessStartInfo start = new ProcessStartInfo {
                FileName = tar,
                Arguments = "-xvf " + Quote(joinedArchive) + " -C " + Quote(baseDirectory),
                WorkingDirectory = baseDirectory,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };
            int extracted = 0;
            StringBuilder errors = new StringBuilder();
            using (Process process = new Process()) {
                process.StartInfo = start;
                process.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e) {
                    if (e.Data == null) return;
                    int current = Interlocked.Increment(ref extracted);
                    if (current == 1 || current % 20 == 0 || current == entryCount) {
                        progress.Report(new InstallUpdate {
                            Percent = 31 + (int)(67L * Math.Min(current, entryCount) / Math.Max(1, entryCount)),
                            Stage = "正在解压完整离线环境",
                            Detail = current + " / " + entryCount + " · " + e.Data
                        });
                    }
                };
                process.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e) { if (e.Data != null && errors.Length < 16000) errors.AppendLine(e.Data); };
                if (!process.Start()) throw new InvalidOperationException("无法启动系统解压程序。");
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                process.WaitForExit();
                if (process.ExitCode != 0) throw new InvalidOperationException("解压失败（tar exit=" + process.ExitCode + "）：" + errors.ToString());
            }

            File.Delete(joinedArchive);
            string launcher = Path.Combine(baseDirectory, rootName, "CCCP-Launcher.exe");
            if (!File.Exists(launcher)) throw new FileNotFoundException("解压完成但未找到 CCCP-Launcher.exe。", launcher);
            return launcher;
        }

        private static string Quote(string value)
        {
            if (value.IndexOf('"') >= 0)
                throw new InvalidDataException("路径不能包含双引号：" + value);
            int trailingBackslashes = 0;
            for (int index = value.Length - 1; index >= 0 && value[index] == '\\'; index--)
                trailingBackslashes++;
            // Windows argv parsing consumes a backslash immediately before the closing
            // quote. Doubling every trailing backslash preserves directory paths such
            // as AppDomain.CurrentDomain.BaseDirectory, which always ends in '\\'.
            return "\"" + value + new string('\\', trailingBackslashes) + "\"";
        }
        private static string FormatBytes(long value) { return (value / 1073741824.0).ToString("0.00") + " GiB"; }
    }

    internal static class Program
    {
        [STAThread]
        private static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new InstallerForm());
        }
    }
}
