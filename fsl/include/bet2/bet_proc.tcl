#

#   bet_proc.tcl - GUI proc for BET - Brain Extraction Tool
#
#   Stephen Smith, FMRIB Image Analysis Group
#
#   Copyright (C) 1999-2000 University of Oxford
#
#   TCLCOPYRIGHT

proc bet_proc { In Out segment_yn overlay_yn mask_yn threshold_yn xtopol_yn cost_yn skull_yn fraction gradient c_x c_y c_z variations InT2} {
    #{{{ setup for running bet 

global PXHOME USER FSLDIR PROCID

    set InF  [ remove_ext $In ]
    set OutF [ remove_ext $Out ]
    if { $InT2 != "" }  { set InT2F  [ remove_ext $InT2 ] } 

#}}}
    #{{{ run command

if { $variations == "-default" } {
    set variations ""
}

if { $variations == "-A2" } {
    set variations "-A2 $InT2F"
}

set thecommand "${FSLDIR}/bin/bet $InF $OutF $variations -f $fraction -g $gradient"

if { $c_x != 0 || $c_y != 0 || $c_z != 0 } {
    set thecommand "${thecommand} -c $c_x $c_y $c_z"
}

if { ! $segment_yn } {
    set thecommand "${thecommand} -n"
}

if { $overlay_yn } {
    set thecommand "${thecommand} -o"
}

if { $mask_yn } {
    set thecommand "${thecommand} -m"
}

if { $threshold_yn } {
    set thecommand "${thecommand} -t"
}

if { $xtopol_yn } {
    set thecommand "${thecommand} -X"
}

if { $cost_yn } {
    set thecommand "${thecommand} -C"
}

if { $skull_yn } {
    set thecommand "${thecommand} -s"
}

puts $thecommand

catch { exec sh -c $thecommand } ErrMsg

puts "$ErrMsg\nFinished"

#}}}
}
