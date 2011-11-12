<?php get_header(); ?>
<div id="container">
	<ul id="nav">
    <li class="<? echo (is_home())?'current_page_item':''; ?>"><a href="<?php echo get_option('home'); ?>/">Home</a></li>
	<?php $pages = wp_list_pages('sort_column=menu_order&depth=1&title_li=&echo=0');
	echo $pages; ?>
    </ul>
	<? unset($pages); ?> 
    <br clear="all" />
    <div id="search"><form method="get" id="searchform" action="<?php bloginfo('url'); ?>/">
    <input type="submit" id="searchsubmit" class="btnSearch" value="&nbsp;" />
    <input type="text" value="<?php the_search_query(); ?>" name="s" id="s" class="txtField" />
    </form></div>
    <div id="site-name"><a href="<?php echo get_option('home'); ?>/"><?php bloginfo('name'); ?></a><br />
    <span class="description"><?php bloginfo('description'); ?></span></div>
    <div class="column01">
    <div id="post-one">
    
<?php if (have_posts()) : while (have_posts()) : the_post(); ?>
		<span class="top"></span>
  		<div class="main-post" id="post-<?php the_ID(); ?>">
    	<h2><a href="<?php the_permalink() ?>" rel="bookmark" title="Permanent Link to <?php the_title(); ?>"><?php the_title(); ?></a></h2>
        <span class="meta">This page created <?php the_time('F') ?> <?php the_time('jS') ?> <?php the_time('Y') ?></span>
        <?php the_content('Read the rest of this entry &raquo;'); ?>
        <br clear="all" />
        </div>
        <span class="btm"></span>
        
<?php endwhile; ?>
<?php else: ?>
 <!-- Error message when no post published -->
<?php endif; ?>         
    </div></div>
<div id="column02">
    	
            <?php 
			$previouspost = get_previous_post($in_same_cat, $excluded_categories);
			if ($previouspost != null) {
			
			echo '<div class="side-post"><div class="upper"><div class="fade"></div><h3>';
			previous_post_link('%link');
			echo '</h3>';
			previous_post_excerpt();
			echo '</div><span class="btn-readon">';
			previous_post_link('%link');
			echo '</span><span class="sub-txt">Previous Entry</span></div>';
        } ?>

            <?php 
			$nextpost = get_next_post($in_same_cat, $excluded_categories);
			if ($nextpost != null) {
			
			echo '<div class="side-post"><div class="upper"><div class="fade"></div><h3>';
			next_post_link('%link');
			echo '</h3>';
			next_post_excerpt();
			echo '</div><span class="btn-readon">';
			next_post_link('%link');
			echo '</span><span class="sub-txt">Next Entry</span></div>';
        } ?>

</div>
    <br clear="all" />
    </div>
    <div class="spacer"></div>
</div>
<div class="lower-outer">
	<div id="lower">
    <?php get_sidebar(); ?>
    </div>
<?php get_footer(); ?>