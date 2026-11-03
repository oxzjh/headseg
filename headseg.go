package headseg

import (
	"errors"
	"unsafe"

	"github.com/oxzjh/cv"
	"github.com/oxzjh/ort"
)

type HeadSeg struct {
	engine *ort.Ort
	width  int
	height int
}

func (h *HeadSeg) Crop(m *cv.Mat, fillWhite bool) (head *cv.Mat, err error) {
	var seg *cv.Mat
	if seg, err = h.extract(m); err != nil {
		return
	}
	mask := seg.CvtColor(cv.COLOR_GRAY2BGR)
	seg.Free()
	fc3 := m.ConvertTo(cv.MatType32FC3)
	head = fc3.Mul(mask, 1)
	if fillWhite {
		revert := mask.ConvertToAB(cv.MatType32FC3, -255, 255)
		head.AddMat(revert)
		revert.Free()
	}
	mask.Free()
	fc3.Free()
	return
}

func (h *HeadSeg) CropFile(file string, fillWhite bool) (*cv.Mat, error) {
	m := cv.IMRead(file, cv.FLAG_COLOR)
	defer m.Free()
	if m.Cols() < 4 || m.Rows() < 4 {
		return nil, errors.New("read file failed")
	}
	return h.Crop(m, fillWhite)
}

func (h *HeadSeg) extract(m *cv.Mat) (seg *cv.Mat, err error) {
	resized := m.Resize(cv.NewSize(h.width, h.height), 0, 0, cv.INTER_LINEAR)
	rgb := resized.CvtColor(cv.COLOR_BGR2RGB)
	resized.Free()
	inputBlob := rgb.ConvertToAB(cv.MatType32FC3, 1/255.0, 0)
	rgb.Free()
	defer inputBlob.Free()
	var result *ort.Result
	if result, err = h.engine.Infer(inputBlob.Data(), inputBlob.Total()*inputBlob.ElemSize(), ort.TypeFloat); err != nil {
		return
	}
	defer result.Free()
	var pointer unsafe.Pointer
	if pointer, err = result.GetOutputPointer(0); err != nil {
		return
	}
	output := cv.NewMatWithData(h.height, h.width, cv.MatType32FC1, pointer)
	seg = output.Resize(cv.NewSize(m.Cols(), m.Rows()), 0, 0, cv.INTER_LINEAR)
	output.Free()
	return
}

func (h *HeadSeg) Close() {
	if h.engine != nil {
		h.engine.Close()
		h.engine = nil
	}
}

func New(model string, threads int) (*HeadSeg, error) {
	engine, err := ort.New(model, "head_seg", threads)
	if err != nil {
		return nil, err
	}
	engine.InputDims[0][0] = 1
	return &HeadSeg{engine, int(engine.InputDims[0][2]), int(engine.InputDims[0][1])}, nil
}
