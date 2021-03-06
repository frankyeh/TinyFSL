#

#   bet.tcl - GUI for BET - Brain Extraction Tool
#
#   Stephen Smith and Matthew Webster, FMRIB Image Analysis Group
#
#   Copyright (C) 1999-2006 University of Oxford
#
#   TCLCOPYRIGHT

#{{{ setups

source [ file dirname [ info script ] ]/fslstart.tcl

set VARS(history) {}

#}}}
#{{{ bet:updatevariations

proc bet:updatevariations { w } {

    global bet

    pack forget $w.f.variations.input

    if { $bet($w,variations) == "-A2" } { 
	pack $w.f.variations.input -in $w.f.variations -side bottom -padx 5 -pady 2 -anchor s
    }
}

#}}}
#{{{ bet:select

proc bet:select { w { dummy "" } } {

    global bet

    set bet($w,input)  [ remove_ext $bet($w,input) ]
    set bet($w,output) [ remove_ext $bet($w,input) ]_brain

#    if { [ string length $bet($w,output) ] == 0 } {
#	set bet($w,output) [ file rootname $bet($w,input) ]_brain
#    }
}

#}}}
#{{{ bet:apply

proc bet:apply { w dialog } {
    global bet

    bet_proc $bet($w,input) $bet($w,output) $bet($w,segment_yn) $bet($w,overlay_yn) $bet($w,mask_yn) $bet($w,threshold_yn) $bet($w,xtopol_yn) $bet($w,cost_yn) $bet($w,skull_yn) $bet($w,fraction) $bet($w,gradient) $bet($w,c_x) $bet($w,c_y) $bet($w,c_z) $bet($w,variations) $bet($w,input2)

    update idletasks

    if {$dialog == "destroy"} {
        bet:destroy $w
    }
}

#}}}
#{{{ bet:destroy

# Summary:      Destroys bet dialog box
proc bet:destroy { w } {
    destroy $w
}

#}}}
#{{{ bet

proc bet { w } {

    #{{{ vars and setup

global bet FSLDIR argc argv PWD

toplevel $w
wm title $w "BET - Brain Extraction Tool - v2.1"
wm iconname $w "BET"
wm iconbitmap $w @${FSLDIR}/tcl/fmrib.xbm

frame $w.f

#}}}
    #{{{ input image

if { $argc > 0 && [ string length [ lindex $argv 0 ] ] > 0 } {
    set inputname [ imglob [ lindex $argv 0 ] ]
    if { [ imtest $inputname ] } {
	if { [ string first / $inputname ] == 0 || [ string first ~ $inputname ] == 0 } {
	    set bet($w,input) $inputname
	} else {
	    set bet($w,input) ${PWD}/$inputname
	}
	set bet($w,output) $bet($w,input)_brain
    }
}


FileEntry $w.f.input -textvariable bet($w,input) -label "Input image   " -title "Select the input image"  -width 50 -filedialog directory  -filetypes IMAGE -command "bet:select $w"
FileEntry $w.f.output -textvariable bet($w,output) -label "Output image" -title "Select the output image"  -width 50 -filedialog directory  -filetypes IMAGE 

#}}}
    #{{{ fractional brain threshold

set bet($w,fraction) 0.5

LabelSpinBox  $w.f.fraction -label "Fractional intensity threshold; smaller values give larger brain outline estimates" -textvariable bet($w,fraction) -range " 0.0 1.0 0.05 " 

#}}}
    #{{{ variations

frame $w.f.variations

set bet($w,variations) -default

optionMenu2 $w.f.variations.menu bet($w,variations) -command "bet:updatevariations $w" -default "Run standard brain extraction using bet2" -R "Robust brain centre estimation (iterates bet2 several times)" -S "Eye & optic nerve cleanup (can be useful in SIENA)" -B "Bias field & neck cleanup (can be useful in SIENA)" -Z "Improve BET if FOV is very small in Z" -F "Apply to 4D FMRI data" -A "Run bet2 and then betsurf to get additional skull and scalp surfaces" -A2 "As above, when also feeding in non-brain-extracted T2"

FileEntry $w.f.variations.input -textvariable bet($w,input2) -label "Input T2 image   " -title "Select T2 input image"  -width 50 -filedialog directory  -filetypes IMAGE

pack $w.f.variations.menu -in $w.f.variations -side top -anchor w

#}}}
    #{{{ Advanced Options

collapsible frame $w.f.opts -title "Advanced options"
set bet($w,xtopol_yn) 0
set bet($w,cost_yn) 0

#{{{ generate segmented image

frame $w.f.opts.segment

label $w.f.opts.segment.label -text "Output brain-extracted image"

set bet($w,segment_yn) 1
checkbutton $w.f.opts.segment.yn -variable bet($w,segment_yn)

pack $w.f.opts.segment.label $w.f.opts.segment.yn -in $w.f.opts.segment -side left

#}}}
#{{{ generate binary brain image

frame $w.f.opts.mask

label $w.f.opts.mask.label -text "Output binary brain mask image"

set bet($w,mask_yn) 0
checkbutton $w.f.opts.mask.yn -variable bet($w,mask_yn)

pack $w.f.opts.mask.label $w.f.opts.mask.yn -in $w.f.opts.mask -side left

#}}}
#{{{ apply thresholding

frame $w.f.opts.threshold

label $w.f.opts.threshold.label -text "Apply thresholding to brain and mask image"

set bet($w,threshold_yn) 0
checkbutton $w.f.opts.threshold.yn -variable bet($w,threshold_yn)

pack $w.f.opts.threshold.label $w.f.opts.threshold.yn -in $w.f.opts.threshold -side left

#}}}
#{{{ generate skull image

frame $w.f.opts.skull

label $w.f.opts.skull.label -text "Output exterior skull surface image"
set bet($w,skull_yn) 0
checkbutton $w.f.opts.skull.yn -variable bet($w,skull_yn)

pack $w.f.opts.skull.label $w.f.opts.skull.yn -in $w.f.opts.skull -side left

#}}}
#{{{ generate overlay image

frame $w.f.opts.overlay

label $w.f.opts.overlay.label -text "Output brain surface overlaid onto original image"

set bet($w,overlay_yn) 0
checkbutton $w.f.opts.overlay.yn -variable bet($w,overlay_yn)

pack $w.f.opts.overlay.label $w.f.opts.overlay.yn -in $w.f.opts.overlay -side left

#}}}
#{{{ gradient brain threshold

set bet($w,gradient) 0

LabelSpinBox $w.f.opts.gradient -label "Threshold gradient; positive values give larger brain outline at bottom, smaller at top" -textvariable bet($w,gradient) -range " -1.0 1.0 0.05 " 

#}}}
#{{{ co-ordinates for centre of initial brain surface sphere

frame $w.f.opts.c

set bet($w,c_x) 0
LabelSpinBox  $w.f.opts.c_x -label "Coordinates (voxels) for centre of initial brain surface sphere " -textvariable bet($w,c_x) -range " 0.0 10000 0.05 " -width 3
set bet($w,c_y) 0
LabelSpinBox  $w.f.opts.c_y -label " Y " -textvariable bet($w,c_y) -range " 0.0 10000 0.05 " -width 3
set bet($w,c_z) 0
LabelSpinBox  $w.f.opts.c_z -label " Z " -textvariable bet($w,c_z) -range " 0.0 10000 0.05 " -width 3

pack $w.f.opts.c_x $w.f.opts.c_y $w.f.opts.c_z  -in $w.f.opts.c  -side left

#}}}

pack $w.f.opts.segment $w.f.opts.mask $w.f.opts.threshold $w.f.opts.skull $w.f.opts.overlay -in $w.f.opts.b -side top -anchor w -pady 0
pack $w.f.opts.gradient $w.f.opts.c -in $w.f.opts.b -side top -anchor w -pady 5

#}}}

    pack $w.f.input $w.f.output $w.f.fraction $w.f.variations $w.f.opts -in $w.f -side top -padx 5 -pady 2 -anchor w

    #{{{ Button Frame

    frame $w.btns
    frame $w.btns.b -relief raised -borderwidth 1
 
    button $w.apply     -command "bet:apply $w keep" \
        -text "Go" -width 5
    bind $w.apply <Return> {
        [winfo toplevel %W].apply invoke}
 
    button $w.cancel    -command "bet:destroy $w" \
        -text "Exit" -width 5
    bind $w.cancel <Return> {
        [winfo toplevel %W].cancel invoke}
 
    button $w.help -command "FmribWebHelp file: ${FSLDIR}/doc/redirects/bet.html" \
            -text "Help" -width 5
    bind $w.help <Return> {
        [winfo toplevel %W].help invoke}

    pack $w.btns.b -side bottom -fill x
    pack $w.apply $w.cancel $w.help -in $w.btns.b \
        -side left -expand yes -padx 3 -pady 10 -fill y
 
    pack $w.f $w.btns -expand yes -fill both

#}}}
}

#}}}

wm withdraw .
bet .rename
tkwait window .rename

