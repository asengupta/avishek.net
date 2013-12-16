<?php
if ( function_exists('register_sidebar') )
    register_sidebar(array(
        'name' => 'Left Footer',
        'before_widget' => '',
        'after_widget' => '',
        'before_title' => '<h3>',
        'after_title' => '</h3>',
    ));
if ( function_exists('register_sidebar') )
    register_sidebar(array(
        'name' => 'Right Footer',
        'before_widget' => '',
        'after_widget' => '',
        'before_title' => '<h3>',
        'after_title' => '</h3>',
    ));

?>