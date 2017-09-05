using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Grpc.Core;
using System.Threading;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using System.IO;



namespace windows
{
    public partial class ImageForm : Form
    {
        private Channel channel;
        private RI.ImageService.ImageServiceClient client;
        private bool shouldStop = true;
        private Thread loopThread;

        public void GetImageLoop()
        {
            try
            {
                while (!shouldStop)
                {
                    while (!loopThread.IsAlive) ;
                    RI.Image image = client.GetImage(new RI.Null(), deadline: DateTime.UtcNow.AddSeconds(1));
                    byte[] poop = image.Data.ToByteArray();
                    Bitmap panelImage = new Bitmap(image.Width, image.Height,
                        System.Drawing.Imaging.PixelFormat.Format24bppRgb);
                    for (int i = 0; i < image.Height; i++)
                    {
                        for (int j = 0; j < image.Width; j++)
                        {
                            Color shit = Color.FromArgb(poop[(i * image.Width + j) * 3 + 2],
                                poop[(i * image.Width + j) * 3 + 1],
                                poop[(i * image.Width + j) * 3]);
                            panelImage.SetPixel(j, i, shit);
                        }
                    }
                    pictureBox.BackgroundImage = panelImage;
                }
                string dir = Path.GetDirectoryName(Application.StartupPath);
                string filename = Path.Combine(dir, @"..\pu.png");
                Console.WriteLine(filename);
                pictureBox.BackgroundImage = Image.FromFile(filename);
            }
            catch
            {
                shouldStop = true;
                this.Invoke(new Action(() => { MessageBox.Show(this, "Cannot reach server.", "Error"); }));
            }
        }

        public ImageForm()
        {
            InitializeComponent();
        }

        private void ConnectButtonClick(object sender, EventArgs e)
        {
            try
            {
                string ipString = ipTextBox.Text;
                Regex regex = new Regex("\\d+\\.\\d+\\.+\\d+\\.\\d+:\\d+");
                if (regex.IsMatch(ipString))
                {
                    channel = new Channel(ipString, ChannelCredentials.Insecure);
                    client = new RI.ImageService.ImageServiceClient(channel);
                    if (shouldStop)
                    {
                        shouldStop = false;
                        loopThread = new Thread(new ThreadStart(GetImageLoop));
                        loopThread.Start();
                    }
                }
                else
                {
                    MessageBox.Show(this, "What the fuck ip did you gave me?", "IP Error");
                }
            }
            catch { }
        }

        private void DisconnectButtonClick(object sender, EventArgs e)
        {
            shouldStop = true;
            try
            {  
                channel.ShutdownAsync().Wait();
            }
            catch
            {
                Console.WriteLine("gun gun");
            }
        }

        private void ImageFormFormClosing(object sender, EventArgs e)
        {
            shouldStop = true;
            try
            {
                channel.ShutdownAsync().Wait();
            }
            catch
            {
                Console.WriteLine("gun gun");
            }
        }
    }
}
