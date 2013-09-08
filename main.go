package main

import (
	"flag"
	"fmt"
	"github.com/JacksCancer/go-opencl/cl"
	"image"
	_ "image/gif"
	"image/jpeg"
	"image/png"
	"io/ioutil"
	"os"
	"path"
)

func listPlatforms(platforms []*cl.Platform) {
	for _, p := range platforms {
		const indent = "  "

		fmt.Println("Platform:   ", p.Name())
		fmt.Println(indent, "Vendor:     ", p.Vendor())
		fmt.Println(indent, "Profile:    ", p.Profile())
		fmt.Println(indent, "Version:    ", p.Version())
		fmt.Println(indent, "Extensions: ", p.Extensions())

		const indent2 = indent + indent

		devices, err := p.GetDevices(cl.DeviceTypeAll)
		checkFatalError(err)

		for _, d := range devices {
			fmt.Println(indent, "Device:     ", d.Name())
			fmt.Println(indent2, "Vendor:     ", d.Vendor())
			fmt.Println(indent2, "Type:       ", d.Type())
			fmt.Println(indent2, "Profile:    ", d.Profile())
			fmt.Println(indent2, "Version:    ", d.Version())
			fmt.Println(indent2, "OpenCL:     ", d.OpenCLCVersion())
			fmt.Println(indent2, "Extensions: ", d.Extensions())
			fmt.Println(indent2, "Driver Ver: ", d.DriverVersion())
			fmt.Println(indent2, "Global Mem: ", d.GlobalMemSize())
			fmt.Println(indent2, "Cache Line: ", d.GlobalMemCachelineSize())
			fmt.Println(indent2, "Cache Type: ", d.GlobalMemCacheType())
			fmt.Println(indent2, "Unified:    ", d.HostUnifiedMemory())
			fmt.Println(indent2, "Local Mem:  ", d.LocalMemSize())
			fmt.Println(indent2, "Local Type: ", d.LocalMemType())
			fmt.Println(indent2, "Units:      ", d.MaxComputeUnits())
			fmt.Println(indent2, "Group Size: ", d.MaxWorkGroupSize())
			fmt.Println(indent2, "Item Size:  ", d.MaxWorkItemSizes())
			fmt.Println(indent2, "Item Dim:   ", d.MaxWorkItemDimensions())
			fmt.Println(indent2, "Img Width:  ", d.Image2DMaxWidth())
			fmt.Println(indent2, "Img Height: ", d.Image2DMaxHeight())
		}
	}
}

func chooseDevice(platform *cl.Platform, devtype cl.DeviceType) *cl.Device {
	devices, err := platform.GetDevices(cl.DeviceTypeAll)
	checkFatalError(err)

	for _, dev := range devices {
		if dev.Type()&devtype != 0 {
			return dev
		}
	}
	return devices[0]
}

func createContext(platform *cl.Platform, dev *cl.Device) *cl.Context {
	ctx, err := cl.CreateContext([]*cl.Device{dev})
	if err != nil {
		panic(err)
	}
	return ctx
}

func buildProgram(c *cl.Context, dev *cl.Device, filename string) *cl.Program {

	data, err := ioutil.ReadFile(filename)
	checkFatalError(err)

	program, err := c.CreateProgramWithSource([]string{string(data)})
	checkFatalError(err)

	err = program.BuildProgram([]*cl.Device{dev}, "")
	if err != nil {
		fmt.Println(err)
		panic("Build Error")
	}

	return program
}

func findImageEdges(c *cl.Context, dev *cl.Device, queue *cl.CommandQueue, src image.Image) (image.Image, *cl.Event) {
	width := src.Bounds().Dx()
	height := src.Bounds().Dy()

	srcTex, err := c.CreateImage2DFromImage(cl.MemUseHostPtr|cl.MemReadOnly, src)
	checkFatalError(err)
	dstTex, err := c.CreateImageSimple(cl.MemWriteOnly, width, height, cl.ChannelOrderRGBA, cl.ChannelDataTypeUNormInt8, nil)
	checkFatalError(err)

	// program, err := c.NewProgramFromFile("optlens.cl")
	// checkFatalError(err)

	program := buildProgram(c, dev, "edgedetect.cl")

	kernel, err := program.CreateKernel("edgeDetect")
	checkFatalError(err)

	err = kernel.SetArg(0, srcTex)
	checkFatalError(err)

	err = kernel.SetArg(1, dstTex)
	checkFatalError(err)

	// err = kernel.SetArg(2, int32(2))
	// checkFatalError(err)

	const edgeSize = 1

	kev, err := queue.EnqueueNDRangeKernel(kernel, []int{edgeSize, edgeSize}, []int{width - 2*edgeSize, height - 2*edgeSize}, nil, nil)
	checkFatalError(err)

	dst := image.NewRGBA(src.Bounds())

	_, err = queue.EnqueueReadImage(dstTex, true, [3]int{0, 0, 0}, [3]int{width, height, 1}, 0, 0, dst.Pix, nil)
	checkFatalError(err)

	return dst, kev
}

func createBufferFromImage(c *cl.Context, src image.Image) (*cl.MemObject, error) {
	switch m := src.(type) {
	case *image.RGBA:
		return c.CreateBuffer(cl.MemUseHostPtr|cl.MemReadOnly, m.Pix)
	case *image.Gray:
		return c.CreateBuffer(cl.MemUseHostPtr|cl.MemReadOnly, m.Pix)
	}
	return nil, fmt.Errorf("image type not supported yet")
}

func findBufferEdges(c *cl.Context, dev *cl.Device, queue *cl.CommandQueue, src image.Image) (image.Image, *cl.Event) {
	width := src.Bounds().Dx()
	height := src.Bounds().Dy()

	srcBuf, err := createBufferFromImage(c, src)
	checkFatalError(err)
	size := width * height * 4 * 4
	dstBuf, err := c.CreateEmptyBuffer(cl.MemWriteOnly, size)
	checkFatalError(err)

	program := buildProgram(c, dev, "edgedetect.cl")

	kernel, err := program.CreateKernel("bufferEdgeDetect")
	checkFatalError(err)

	err = kernel.SetArg(0, srcBuf)
	checkFatalError(err)

	err = kernel.SetArg(1, dstBuf)
	checkFatalError(err)

	err = kernel.SetArg(2, int32(width))
	checkFatalError(err)

	const edgeSize = 1

	kev, err := queue.EnqueueNDRangeKernel(kernel, []int{edgeSize, edgeSize}, []int{width - 2*edgeSize, height - 2*edgeSize}, nil, nil)
	checkFatalError(err)

	dst := image.NewRGBA(src.Bounds())

	_, err = queue.EnqueueReadBufferUint8(dstBuf, true, 0, dst.Pix, nil)
	checkFatalError(err)

	return dst, kev
}

func findEdges(src image.Image, devtype cl.DeviceType, useBuffer bool) image.Image {
	platforms, err := cl.GetPlatforms()
	checkFatalError(err)
	listPlatforms(platforms)
	platform := platforms[0]
	dev := chooseDevice(platform, devtype)
	c := createContext(platform, dev)
	queue, err := c.CreateCommandQueue(dev, cl.CommandQueueProfilingEnable)
	checkFatalError(err)

	var (
		kev *cl.Event
		dst image.Image
	)

	if dev.ImageSupport() && !useBuffer {
		dst, kev = findImageEdges(c, dev, queue, src)
	} else {
		if !useBuffer {
			fmt.Printf("No image support on device %v\n", dev.Name())
		}
		dst, kev = findBufferEdges(c, dev, queue, src)
	}

	start, err := kev.GetEventProfilingInfo(cl.ProfilingInfoCommandStart)
	checkFatalError(err)
	end, err := kev.GetEventProfilingInfo(cl.ProfilingInfoCommandEnd)
	checkFatalError(err)

	fmt.Printf("time: %vms, (res: %v)\n", float64(end-start)/1000000., dev.ProfilingTimerResolution())

	return dst
}

func checkError(err error) {
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func checkFatalError(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {

	var (
		srcFile   string
		dstFile   string
		useGpu    bool
		useBuffer bool
	)

	flag.StringVar(&srcFile, "src", "", "source image")
	flag.StringVar(&dstFile, "dst", "out.png", "destination image")
	flag.BoolVar(&useGpu, "gpu", false, "use gpu")
	flag.BoolVar(&useBuffer, "buffer", false, "use buffer instead of images")

	flag.Parse()

	if len(srcFile) == 0 {
		flag.PrintDefaults()
		os.Exit(1)
	}

	src, err := os.Open(srcFile)
	checkError(err)

	dst, err := os.Create(dstFile)
	checkError(err)

	srcImg, _, err := image.Decode(src)
	checkError(err)

	devType := cl.DeviceTypeCPU
	if useGpu {
		devType = cl.DeviceTypeGPU
	}

	dstImg := findEdges(srcImg, devType, useBuffer)

	switch path.Ext(dstFile) {
	case "jpeg":
		err = jpeg.Encode(dst, dstImg, nil)
		checkFatalError(err)
	default:
		err = png.Encode(dst, dstImg)
		checkFatalError(err)
	}
}
