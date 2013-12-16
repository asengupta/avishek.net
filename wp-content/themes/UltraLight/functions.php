<?php
if ( function_exists('register_sidebar') )
    register_sidebar(array(
        'before_widget' => '<div class="widget">',
        'after_widget' => '</div>',
        'before_title' => '<h2>',
        'after_title' => '</h2>',
    ));
?>
