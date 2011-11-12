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
    <div class="spacer"></div>
    <div id="recent-posts">
    <!-- post begin -->
  <?php if (have_posts()) : while (have_posts()) : the_post(); ?>
  <div class="home-post" id="post-<?php the_ID(); ?>">
        	<div class="upper">
            <div class="fade"></div>
            <h3><a href="<?php the_permalink() ?>" rel="bookmark" title="Permanent Link to <?php the_title(); ?>"><?php the_title(); ?></a></h3>
            <span class="meta"><?php the_time('F') ?> <?php the_time('jS') ?> <?php the_time('Y') ?></span>
            <?php the_excerpt('Read the rest of this entry &raquo;'); ?>
            </div>
            <span class="btn-readon"><a href="<?php the_permalink() ?>" rel="bookmark" title="Permanent Link to <?php the_title(); ?>">Read On</a></span>
            <span class="lower-meta"><?php comments_popup_link('No Comments', '1 Comment', '% Comments'); ?></span>
        </div>
<?php endwhile; ?>
<br clear="all" />
<?php else: ?>
<div class="search-results"><span class="bigger">Search Results...</span><br /><?php _e('Sorry, no posts matched your criteria.'); ?></div>
<?php endif; ?>        
    <!-- post end -->
<div id="page-nav"><span class="older"><?php next_posts_link('Older Entries') ?></span><span class="newer"><?php previous_posts_link('Newer Entries') ?></span></div>
    <br clear="all" />
    </div>
</div>
<div class="lower-outer">
	<div id="lower">
    <?php get_sidebar(); ?>
    </div>
<?php get_footer(); ?>